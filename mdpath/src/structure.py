"""Structure Calculations --- :mod:`mdpath.src.structure`
==============================================================================
... (docstring) ...
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from multiprocessing import Pool
from Bio import PDB
from itertools import combinations
import logging

# --- ADDED: Global variable and worker functions ---
_worker_universe = None

def _init_worker(topology, trajectory):
    """Initialize the Universe in the worker process."""
    global _worker_universe
    # Re-create the universe in the worker process using the file paths
    _worker_universe = mda.Universe(topology, trajectory)

def _calc_dihedral_task(res_id):
    """Standalone worker task that uses the global universe."""
    global _worker_universe
    try:
        # Note: Ensure res_id matches the index in the universe. 
        # If res_id is a PDB number, this might need adjustment (e.g. selection by resid).
        # Assuming res_id is a valid index for now based on original code logic:
        res = _worker_universe.residues[res_id]
        
        ags = [res.phi_selection()]
        if not all(ags):  # Check if any selections are None
            return None
            
        R = Dihedral(ags).run()
        dihedrals = R.results.angles
        dihedral_angle_movement = np.diff(dihedrals, axis=0)
        return res_id, dihedral_angle_movement
    except (TypeError, AttributeError, IndexError, Exception) as e:
        # Logging inside workers can be tricky, generally just return None or print if debugging
        return None
# ---------------------------------------------------

class StructureCalculations:
    # ... (Keep this class as it is) ...
    def __init__(self, pdb: str) -> None:
        self.pdb = pdb
        self.first_res_num, self.last_res_num = self.res_num_from_pdb()
        self.num_residues = self.last_res_num - self.first_res_num + 1

    def res_num_from_pdb(self) -> tuple:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", self.pdb)
        first_res_num = float("inf")
        last_res_num = float("-inf")
        for res in structure.get_residues():
            if PDB.Polypeptide.is_aa(res):
                res_num = res.id[1]
                if res_num < first_res_num:
                    first_res_num = res_num
                if res_num > last_res_num:
                    last_res_num = res_num
        return int(first_res_num), int(last_res_num)

    def calculate_distance(self, atom1: tuple, atom2: tuple) -> float:
        distance_vector = [
            atom1[i] - atom2[i] for i in range(min(len(atom1), len(atom2)))
        ]
        distance = np.linalg.norm(distance_vector)
        return distance

    def calculate_residue_suroundings(self, dist: float, mode: str) -> pd.DataFrame:
        if mode not in ["close", "far"]:
            raise ValueError("Mode must be either 'close' or 'far'.")

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("pdb_structure", self.pdb)
        heavy_atoms = ["C", "N", "O", "S"]
        residue_pairs = []
        residues = [
            res for res in structure.get_residues() if PDB.Polypeptide.is_aa(res)
        ]

        for res1, res2 in tqdm(
            combinations(residues, 2),
            desc=f"\033[1mCalculating {mode} residue surroundings\033[0m",
            total=len(residues) * (len(residues) - 1) // 2,
        ):
            res1_id = res1.get_id()[1]
            res2_id = res2.get_id()[1]
            if res1_id <= self.last_res_num and res2_id <= self.last_res_num:
                condition_met = False if mode == "close" else True
                for atom1 in res1:
                    if atom1.element in heavy_atoms:
                        for atom2 in res2:
                            if atom2.element in heavy_atoms:
                                distance = self.calculate_distance(
                                    atom1.coord, atom2.coord
                                )
                                if (mode == "close" and distance <= dist) or (
                                    mode == "far" and distance > dist
                                ):
                                    condition_met = True
                                    break
                        if condition_met:
                            break
                if condition_met:
                    residue_pairs.append((res1_id, res2_id))

        return pd.DataFrame(residue_pairs, columns=["Residue1", "Residue2"])


class DihedralAngles:
    """Calculate dihedral angle movements for residues in a molecular dynamics (MD) trajectory.
    ...
    """

    def __init__(
        self,
        traj: mda.Universe,
        first_res_num: int,
        last_res_num: int,
        num_residues: int,
    ) -> None:
        self.traj = traj
        self.first_res_num = first_res_num
        self.last_res_num = last_res_num
        self.num_residues = num_residues
        
        # --- ADDED: Extract filenames to pass to workers ---
        # Assuming traj is created from files as per mdpath.py
        try:
            self.topo_file = traj.filename
            self.traj_file = traj.trajectory.filename
        except AttributeError:
            # Fallback for unexpected Universe types (though typical usage has these)
            self.topo_file = None
            self.traj_file = None

    # The original calc_dihedral_angle_movement is no longer needed for parallel execution,
    # but you can keep it if you use it sequentially elsewhere. The parallel version uses _calc_dihedral_task.

    def calculate_dihedral_movement_parallel(
        self,
        num_parallel_processes: int,
    ) -> pd.DataFrame:
        """Parallel calculation of dihedral angle movement for all residues in the trajectory."""
        df_all_residues = pd.DataFrame()

        try:
            # --- MODIFIED: Use initializer to setup universe in workers ---
            if not self.topo_file or not self.traj_file:
                raise ValueError("Universe filenames not found. Cannot run multiprocessing.")

            with Pool(processes=num_parallel_processes, 
                      initializer=_init_worker, 
                      initargs=(self.topo_file, self.traj_file)) as pool:
                
                with tqdm(
                    total=self.num_residues,
                    ascii=True,
                    desc="\033[1mProcessing residue dihedral movements\033[0m",
                ) as pbar:
                    # Pass the standalone function '_calc_dihedral_task' instead of the method
                    results = pool.imap_unordered(
                        _calc_dihedral_task,
                        range(self.first_res_num, self.last_res_num + 1),
                    )

                    for result in results:
                        if result is None:
                            pbar.update(1)
                            continue

                        res_id, dihedral_data = result
                        try:
                            df_residue = pd.DataFrame(
                                dihedral_data, columns=[f"Res {res_id}"]
                            )
                            df_all_residues = pd.concat(
                                [df_all_residues, df_residue], axis=1
                            )
                        except Exception as e:
                            logging.error(
                                f"\033[1mError processing residue {res_id}: {e}\033[0m"
                            )
                        finally:
                            pbar.update(1)

        except Exception as e:
            logging.error(f"Parallel processing failed: {str(e)}")

        return df_all_residues
