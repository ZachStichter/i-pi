# this is how the driver will be referred to in the input files
__DRIVER_NAME__ = "qchem"
__DRIVER_CLASS__ = "QChem_driver"

_VALID_QCHEM_VERSIONS = ["6.1.0"]

import json
from .dummy import Dummy_driver
from ipi.utils.messages import info, warning, verbosity
from ipi.utils.softexit import softexit
from os.path import abspath, join
from os import makedirs, environ
from ipi.utils.units import unit_to_internal, unit_to_user
import numpy as np
from numpy.typing import ArrayLike
import subprocess
import re

class SCFError(ValueError):
    """Dummy error to notify the code that an SCF attempt hasn't converged."""
    pass


class QChem_driver(Dummy_driver):
    """
    Base class providing the structure of a PES for the python driver.

    Command line:
        i-pi-py_driver -m qchem -o scratchdir=.,job=qchem_input_file.in,qchem_cores=1,apt_processes=1
    Init arguments:
        :param verbose: bool to determine whether the PES should output verbose info
        :param scratchdir: string, path to the target scratch directory. QChem input and output files will be created here.
        :param job: string, QChem-style input file. Include $molecule and $rem blocks. Net charge, multiplicity, and atom identities will be read from the $molecule block. If using CavPh, do not include photons in this block. Automatically sets symmetry to False in the $rem block. Does not support Z-matrix notation.
        :qchem_cores: int, Number of cores per QChem process
        :dipole_derivative: bool, calculate the spatial derivative of the electronic dipole using the finite field method
        :apt_processes: int, number of dipole derivative components to calculate in parallel (only used if dipole_derivative is True)
        :basisfile: string, path to QChem basis file
        :ecpfile: string, path to QChem ECP file
    """

    def __init__(self, scratchdir=".", job=None, nslots=0, dipole_derivative=False, apt_processes=1, basisfile=None, ecpfile=None, *args, **kwargs):
        info(f'Starting QChem Driver. Tested QChem versions include {_VALID_QCHEM_VERSIONS}.\nNote that this driver CANNOT perform NPT or NPH simulations, as it does not calculate the virial.', verbosity.low)
        self.scratchdir = scratchdir
        self.job = job
        self.nslots = nslots
        self.dipole_derivative = dipole_derivative
        self.apt_processes = apt_processes
        self.basisfile = basisfile
        self.ecpfile = ecpfile
        self.inputfile = None
        self.outputfile = None
        self.results = {}
        self.fd_delta = 5e-4

        super().__init__(*args, **kwargs)

    def check_parameters(self):
        """Verify parameters are valid. Loads the QChem template and sets up the dipole derivative calculation if used."""
        global _VALID_QCHEM_VERSIONS
        self._prep_input_directory()
        try:
            ver = subprocess.run(["qchem"], capture_output=True, text=True, env=environ.copy())
            ver = ver.stdout + ver.stderr
        except FileNotFoundError:
            raise ValueError("QChem is not available on this system")
        ver = re.search(r"Q-Chem version:\s*([\d\.]+)", ver, re.IGNORECASE)
        if ver is None:
            raise ValueError("QChem is not available on this system")

        if ver.group(1) not in _VALID_QCHEM_VERSIONS:
            warning(f"Untested QChem version {ver.group(1) if ver else '[Unknown]'}. This driver has only been tested with QChem version(s) {_VALID_QCHEM_VERSIONS}. This driver may fail if QChem output file format changes. If you find anomolous errors, try a tested version of QChem first, and consider adding an interface for your version. If everything works as expected, please add your QChem version to the list of valid QChem versions at the top of /pes/qchem.py.")
        self._read_qchem_template()

    def compute_structure(self, cell, pos):
        """Evaluate a single structure"""
        pos_ang = unit_to_user("length", "angstrom", pos)
        pot, force, electric_dipole = self._calculate_force(cell, pos_ang)
        # just evaluates zeros 
        vir = cell * 0.0  # makes a zero virial with same shape as cell

        print(self.dipole_derivative)

        if self.dipole_derivative:
            dipder = self._calculate_apt(cell, pos_ang)
            print(dipder)
            extras = json.dumps(
                {"dipole": electric_dipole.tolist(),
                 "dipole_derivative": dipder.flatten().tolist(),
                }
            )
        else:
            extras = json.dumps(
                {"dipole": electric_dipole.tolist(),
                 "dipole_derivative": np.zeros(9*self.num_atoms),
                }
            )
        return pot, force, vir, extras

    def compute(self, cell, pos):
        """Does nothing, but returns properties that can be used by the driver loop."""

        if isinstance(cell, list):
            if not isinstance(pos, list) or len(cell) != len(pos):
                raise ValueError(
                    "Both position and cell should be given as lists to run in batched mode"
                )
            warning("Batched execution will execute in serial.", verbosity.high)
            return [self.compute_structure(cell, pos) for cell, pos in zip(cell, pos)]
        else:
            return self.compute_structure(cell, pos)

    def __call__(self, cell, pos):
        """Function interface"""

        return self.compute(cell, pos)
    
    def _read_qchem_template(self):
        """Loads the QChem template into memory"""
        if not self.job:
            raise ValueError("`job` parameter is not present. Correct this and try again.")
        try:
            with open(self.job, "r") as i:
                content = i.readlines()
        except FileNotFoundError:
            raise ValueError(f"`job` file is not found at {self.job}. Correct this and try again.")
        except Exception as e:
            raise ValueError(f"Unknown error occurred on loading QChem job file: {e}")
        
        self.molecule_block = []
        self.rem_block = []
        self.num_atoms = 0
        method_flag = False
        basis_flag = False
        sym_flag = False
        scf_algorithm_flag = False
        jobtype_flag = False
        multiplicity_flag = False
        in_block = None
        for line in content:
            fline = line.strip()
            if in_block:
                match in_block:
                    case "mol":
                        if fline.upper().startswith("$END"):
                            multiplicity_flag = False
                            self.molecule_block.append([fline])
                            in_block = None
                        elif not multiplicity_flag:
                            self.molecule_block.append([fline])
                            multiplicity_flag = True
                        elif fline and not fline.startswith("!"):
                            parts = fline.split()
                            self.molecule_block.append([parts[0]]+parts[4:])
                            self.num_atoms += 1 
                        else:
                            self.molecule_block.append([fline])
                    case "rem":
                        if fline.upper().startswith("$END"):
                            if not method_flag:
                                self.rem_block.append("METHOD HF")
                            if not basis_flag:
                                self.rem_block.append("BASIS 6-31G")
                            if not sym_flag:
                                self.rem_block.append("SYM_IGNORE TRUE")
                            if not jobtype_flag:
                                self.rem_block.append("JOBTYPE FORCE")
                            if not scf_algorithm_flag:
                                self.rem_block.append("SCF_ALGORITHM DIIS")
                            method_flag = False
                            basis_flag = False
                            sym_flag = False
                            scf_algorithm_flag = False
                            in_block = None
                        elif fline.upper().startswith("METHOD"):
                            method_flag = True
                        elif fline.upper().startswith("BASIS"):
                            basis_flag = True
                        elif fline.upper().startswith("SCF_ALGORITHM"):
                            scf_algorithm_flag = True
                        elif fline.upper().startswith("SYM_IGNORE"):
                            parts = fline.replace("="," ").split()
                            if "FALSE" in parts:
                                warning("QChem $rem block contains explicit symmetry tracking. This is incompatible with MD. Overwriting.")
                                fline = "SYM_IGNORE TRUE"
                            sym_flag = True
                        elif fline.upper().startswith("JOBTYPE"):
                            parts = fline.replace("=", " ").split()
                            if parts[1] not in ["FORCE", "SP"]:
                                warning("QChem job type is incorrect. Defaulting to force calculation.")
                                fline = "JOBTYPE FORCE"
                            jobtype_flag = True
                        self.rem_block.append(fline.replace("=", " "))

            elif fline.upper().startswith("$MOLECULE"):
                self.molecule_block.append([fline])
                in_block = 'mol'
            elif fline.upper().startswith("$REM"):
                self.rem_block.append(fline)
                in_block = 'rem'

    def _format_molecule_block(self, pos):
        pos3 = pos.reshape(-1, 3)

        symbols = [line[0] for line in self.molecule_block[2:-1]]
        if self.num_atoms != pos3.shape[0]:
            raise ValueError(f"Molecule block of atoms ({self.num_atoms} atom(s): {symbols}) does not equal shape of I-Pi positions {pos3.shape[0]}.")
        molecule_block = [line[0] for line in self.molecule_block[:2]] # this is super janky because I extracted self.molecule_block into a 2-d list to preserve possible special formatting from qchem input file
        atom_idx = 0 
        for atom in self.molecule_block[2:-1]:
            clean_atom = atom[0].strip()
            if not clean_atom:
                continue
            if not clean_atom.startswith("!"):
                if len(atom) > 1:
                    astring = f"{atom[0]} {pos3[atom_idx,0]} {pos3[atom_idx,1]} {pos3[atom_idx,2]} {atom[1:]}"
                else:
                    astring = f"{atom[0]} {pos3[atom_idx,0]} {pos3[atom_idx, 1]} {pos3[atom_idx, 2]}"
                molecule_block.append(astring)
                atom_idx += 1
        molecule_block.append(self.molecule_block[-1][0])
        molecule_block = "\n".join(molecule_block)
        molecule_block += "\n"
        return molecule_block
    
    def _format_rem_block(self):
        return "\n".join(self.rem_block)+"\n"
    
    def _format_basis_file(self):
        if self.basisfile is None:
            return "\n"
        with open(self.basisfile) as i:
            basis = i.readlines()
        return "\n".join(["$basis"]+basis+["$end"])+"\n"
    
    def _format_ecp_file(self):
        if self.ecpfile is None:
            return "\n"
        with open(self.ecpfile) as i:
            ecp = i.readlines()
        return "\n".join(["$ecp"]+ecp+["$end"])+"\n"

    def _format_multipole_field(self, Ex=0, Ey=0, Ez=0):
        if Ex!=0 or Ey!=0 or Ez!=0:
            return f"$MULTIPOLE_FIELD\nX {Ex}\nY {Ey}\nZ {Ez}\n$END\n"
        else:
            return "\n"

    def _prep_input_directory(self):
        # TODO: look into making this definition threadsafe. Currently there may be overlaps in parallel calls to the same driver at the same location, leading to race conditions and collisions in file io.
        absdir = abspath(self.scratchdir)
        makedirs(absdir, exist_ok=True)
        self.inputfile = join(absdir, "qchem.in")
        self.outputfile = join(absdir, "qchem.out")

        # reserve space for finite field calculation (won't be used some runs)
        self.finite_in_plus_x = join(absdir, "ffpx.in")
        self.finite_in_plus_y = join(absdir, "ffpy.in")
        self.finite_in_plus_z = join(absdir, "ffpz.in")
        self.finite_in_minus_x = join(absdir, "ffmx.in")
        self.finite_in_minus_y = join(absdir, "ffmy.in")
        self.finite_in_minus_z = join(absdir, "ffmz.in")
        self.finite_out_plus_x = join(absdir, "ffpx.out")
        self.finite_out_plus_y = join(absdir, "ffpy.out")
        self.finite_out_plus_z = join(absdir, "ffpz.out")
        self.finite_out_minus_x = join(absdir, "ffmx.out")
        self.finite_out_minus_y = join(absdir, "ffmy.out")
        self.finite_out_minus_z = join(absdir, "ffmz.out")

        self.fd_mapping = {(0,1):(self.finite_in_plus_x,self.finite_out_plus_x),
                           (1,1):(self.finite_in_plus_y,self.finite_out_plus_y),
                           (2,1):(self.finite_in_plus_z,self.finite_out_plus_z),
                           (0,-1):(self.finite_in_minus_x,self.finite_out_minus_x),
                           (1,-1):(self.finite_in_minus_y,self.finite_out_minus_y),
                           (2,-1):(self.finite_in_minus_z,self.finite_out_minus_z)}

    def _write_qchem_input(self, cell, pos, file, multipole_fields=(0,0,0)):
        """
        multipole_fields is a 3-tuple containing the x-, y-, and z- components of the electric field for unpacking later.
        """
        if file is None:
            self._prep_input_directory()
        with open(file, "w+") as o:
            o.write(self._format_molecule_block(pos))
            o.write(self._format_rem_block())
            o.write(self._format_basis_file())
            o.write(self._format_ecp_file())
            o.write(self._format_multipole_field(*multipole_fields))

    def _read_qchem_output(self, file_contents):
        """
        Parse QChem output from file contents. Returns energy, force, dipole

        Args:
            file_contents (str): a list containing the lines extracted from a QChem output file. Pass a list equivalent to thefile.readlines()
            
        Returns:
            A tuple containing the energy, forces, and electric dipole
        """
        
        gradient_flag = False
        dipole_flag = False
        skip_next_line = False
        energy = 0
        scf_gradient = np.zeros((3, self.num_atoms))
        scf_dipole = np.zeros(3)
        gradient_atom = 0
        for line in file_contents:
            if 'SCF failed to converge' in line:
                raise SCFError("SCF Failed to Converge.")
            elif 'ERROR: alpha_min' in line:
                raise SCFError("Error: alpha_min")
            elif ' Total energy in the final basis set =' in line:
                energy = float(line.split("=")[-1].strip())
            elif ' Gradient of SCF Energy' in line:
                gradient_flag = True
                skip_next_line = True
            elif ' Max gradient component' in line:
                gradient_flag = False # this is the end of the gradient block
            elif '   Dipole Moment (Debye)' in line:
                dipole_flag = True
            elif gradient_flag and skip_next_line:
                skip_next_line = False
            elif gradient_flag:
                left_margin = line[:6].strip()
                if not left_margin:
                    continue # skip the block index header
                else:
                    col_head = line[:6] # split the coords line
                    fline = line[6:]
                coord_idx = int(col_head)-1 # 1->x, 2->y, 3->z
                chunk_size = 12
                grad_vals = [
                        float(fline[i : i + chunk_size].strip())
                        for i in range(0, len(fline), chunk_size)
                        if fline[i : i + chunk_size].strip() # Safeguard against trailing whitespace
                    ]
                natoms = len(grad_vals)
                scf_gradient[coord_idx,gradient_atom:gradient_atom+natoms] = grad_vals
                if coord_idx >=2:
                    gradient_atom += natoms
            elif dipole_flag:
                fline = line.strip().split()
                scf_dipole[:] = [float(val) for val in line.strip().split()[1::2]]
                dipole_flag = False
        en = unit_to_internal("energy", "atomic_unit", energy)
        fc = -unit_to_internal("force", "atomic_unit", scf_gradient).T # F = -grad E; transpose to give columns (#,coord) format instead of rows (coord, #)
        dp = unit_to_internal("electric-dipole", "debye", scf_dipole)
        return en, fc, dp

    def _calculate_apt(self, cell, pos)->ArrayLike:
        if not self.dipole_derivative:
            return np.zeros((self.num_atoms,3,3))
        else:
            self.old_rem = self.rem_block.copy()
            self.rem_block.insert(-1,"MULTIPOLE_FIELD TRUE")
            pending_jobs = []
            fd_grads = {}
            # x, y, z
            for axis in range(3):
                for job in [1,-1]:
                    job_id = (axis, job)
                    infile, outfile = self.fd_mapping[job_id]
                    field_tuple = np.zeros(3)
                    field_tuple[axis] = job*self.fd_delta
                    field_tuple = tuple(field_tuple)
                    pending_jobs.append((job_id, infile, outfile, field_tuple))
            for batch_start_idx in range(0, len(pending_jobs), self.apt_processes):
                batch = pending_jobs[batch_start_idx:batch_start_idx+self.apt_processes]
                running_jobs = []
                for job_id, infile, outfile, field_tuple in batch:
                    self._write_qchem_input(cell, pos, infile, field_tuple)
                    with open(outfile, 'w'): pass # just overwrite any old file to clear stale data
                    proc = subprocess.Popen(["qchem", infile, outfile], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    running_jobs.append((job_id, infile, outfile, proc))
                for job_id, infile, outfile, proc in running_jobs:
                    proc.wait()
                    with open(outfile, "r") as result:
                        output_content = result.readlines()
                    try:
                        _, forces, _ = self._read_qchem_output(output_content)
                    except SCFError:
                        forces = np.zeros(pos.shape, dtype=float)
                        with open(infile, "r") as file:
                            content = file.read()
                        updated_content = re.sub(r"(SCF_ALGORITHM\s*=\s*|\s+)DIIS", r"\1GDM", content, flags=re.IGNORECASE)
                        with open(infile, "w") as file:
                            file.write(updated_content)
                        with open(outfile, 'w'): pass # overwrite old file to clear stale data
                        subprocess.run(["qchem", infile, outfile], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        try:
                            with open(outfile, "r") as result:
                                output_content = result.readlines()
                            _, forces, _ = self._read_qchem_output(output_content)
                        except SCFError:
                            raise ValueError("Cannot converge SCF")
                    forces = forces.flatten()
                    fd_grads[job_id] = forces
            dipder = np.zeros((self.num_atoms, 3, 3))
            for i in range(3):
                grads_i = -(fd_grads[(i,1)]-fd_grads[(i,-1)])/(2*self.fd_delta)
                dipder[:,:,i] = grads_i.reshape(self.num_atoms, 3)
            self.rem_block = self.old_rem
            return dipder
        
    def _calculate_force(self, cell, pos):
        self._write_qchem_input(cell, pos, self.inputfile)
        with open(self.outputfile, 'w'): pass # clear old output file
        subprocess.run(["qchem", self.inputfile, self.outputfile], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(self.outputfile, "r") as outfile:
            result = outfile.readlines()
        try:
            en, force, dipole = self._read_qchem_output(result)
        except SCFError:
            with open(self.inputfile, "r") as file:
                content = file.read()
            updated_content = re.sub(r"(SCF_ALGORITHM\s+)DIIS", r"\1GDM", content, flags=re.IGNORECASE)
            with open(self.inputfile, "w") as file:
                file.write(updated_content)
            with open(self.outputfile, 'w'): pass # clear old output file
            subprocess.run(["qchem", self.inputfile, self.outputfile], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(self.outputfile, "r") as outfile:
                result = outfile.readlines()
            try:
                en, force, dipole = self._read_qchem_output(result)
            except SCFError:
                force = np.empty(pos.shape)
                force.fill(np.nan)
                en = np.nan
                dipole = np.empty((3))
                dipole.fill(np.nan)
        return en, force, dipole

r"""gemini refactor:

def _calculate_force(self, cell, pos):
    # 1. Generate the initial input file (defaults to DIIS)
    self._write_qchem_input(cell, pos, self.inputfile)
    
    # Define our execution attempts: False (use DIIS), True (switch to GDM)
    for switch_to_gdm in [False, True]:
        if switch_to_gdm:
            # Modify the file in-place if the first attempt failed
            with open(self.inputfile, "r") as file:
                content = file.read()
            updated_content = re.sub(r"(SCF_ALGORITHM\s+)DIIS", r"\1GDM", content, flags=re.IGNORECASE)
            with open(self.inputfile, "w") as file:
                file.write(updated_content)
        
        # 2. Run Q-Chem
        subprocess.run(["qchem", self.inputfile, self.outputfile])
        
        # 3. Read output
        with open(self.outputfile, "r") as outfile:
            output_lines = outfile.readlines()
            
        # 4. Attempt to parse
        try:
            # If successful, return immediately and exit the function
            return self._read_qchem_output(output_lines)
        except SCFError:
            # If even the GDM fallback failed, we are out of options
            if switch_to_gdm:
                return np.full(pos.shape, np.nan)

probably fix later
"""
