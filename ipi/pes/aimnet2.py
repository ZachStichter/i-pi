"""
AIMNet2 PES Driver for i-PI

Connects the AIMNet2 neural network potential to i-PI using the ASE driver base class.
"""

from ipi.pes._ase import ASEDriver

__DRIVER_NAME__ = "aimnet2"
__DRIVER_CLASS__ = "AIMNet2Driver"


class AIMNet2Driver(ASEDriver):
    """
    i-PI driver for AIMNet2 via the ASE calculator interface.

    example: i-pi-py_driver -m aimnet2 -u -o template.xyz,model='aimnet2',charge=0.0,mult=0,lrcoulomb_method='ewald' -S /path/to/driver/prefix

    Parameters:
        template (str): Path to an ASE-readable structure file (e.g., .xyz, .pdb, .cif).
        model (str): AIMNet2 model name or model path (e.g., 'aimnet2', 'isayevlab/aimnet2-2025', 'aimnet2_wb97m_d3').
        charge (float): Total molecular/system charge (default: 0.0).
        mult (int): Spin multiplicity (default: 1).
        lrcoulomb_method (str): Long-range Coulomb method for periodic systems ('dsf', 'ewald', or None).
        has_energy (bool): Compute energy (default: True).
        has_forces (bool): Compute forces (default: True).
        has_stress (bool): Compute stress tensor for periodic systems (default: True).
    """

    def __init__(
        self,
        template,
        model="aimnet2",
        charge=0.0,
        mult=1,
        lrcoulomb_method=None,
        has_energy=True,
        has_forces=True,
        has_stress=False,
        *args,
        **kwargs,
    ):
        self.model_name = model
        self.charge = charge
        self.mult = mult
        self.lrcoulomb_method = lrcoulomb_method

        # Initialize base ASEDriver
        super().__init__(
            template=template,
            has_energy=has_energy,
            has_forces=has_forces,
            has_stress=has_stress,
            *args,
            **kwargs,
        )

    def check_parameters(self):
        """Loads template structure and instantiates the AIMNet2 ASE calculator."""
        # Reads the atomic structure template and sets self.template_ase
        super().check_parameters()

        # Import AIMNet2 ASE Calculator interface
        try:
            from aimnet.calculators import AIMNet2ASE, AIMNet2Calculator
        except ImportError:
            try:
                from aimnet2calc import AIMNet2ASE, AIMNet2Calculator
            except ImportError:
                raise ImportError(
                    "Could not import AIMNet2 ASE interface. "
                    "Please install aimnet via: pip install 'aimnet[ase]'"
                )

        class PatchedAIMNet2(AIMNet2Calculator):
            def calculate(self, atoms=None, properties=None, system_changes=None):
                super().calculate(atoms, properties, system_changes)
                
                # Intercept the results dictionary and rename the key
                if 'dipole_moment' in self.results:
                    self.results['dipole'] = self.results.pop('dipole_moment')

        # Initialize the calculator with model parameters
        self.ase_calculator = AIMNet2ASE(
            base_calc=PatchedAIMNet2(self.model_name, compile_model=True),
            charge=self.charge,
            mult=self.mult,
        )

        # Configure long-range Coulomb settings if specified for periodic systems
        if self.lrcoulomb_method is not None:
            if hasattr(self.ase_calculator, "set_lrcoulomb_method"):
                self.ase_calculator.set_lrcoulomb_method(self.lrcoulomb_method)
            elif hasattr(self.ase_calculator, "calc") and hasattr(
                self.ase_calculator.calc, "set_lrcoulomb_method"
            ):
                self.ase_calculator.calc.set_lrcoulomb_method(self.lrcoulomb_method)