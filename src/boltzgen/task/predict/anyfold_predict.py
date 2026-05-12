"""
AnyFold integration for BoltzGen
Simple, direct usage of AnyFold's CofoldingTask
"""

import os
import json
import tempfile
from pathlib import Path
from omegaconf import OmegaConf

from boltzgen.task.task import Task

# AnyFold imports
try:
    from anyfold.inference.cofolding import CofoldingTask
    ANYFOLD_AVAILABLE = True
except ImportError:
    ANYFOLD_AVAILABLE = False
    print("Warning: AnyFold not available. Install anyfold_dev to use this feature.")


class AnyFoldPredict(Task):
    """BoltzGen task that uses AnyFold for structure prediction"""

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs

        if not ANYFOLD_AVAILABLE:
            raise ImportError("AnyFold is not available")

    def run(self, config):
        """Run AnyFold prediction on BoltzGen structures"""

        # Read BoltzGen intermediate structures - use input_dir (full path) not input (relative path)
        input_dir = config.get("input_dir", config.get("input", "intermediate_designs_inverse_folded"))

        # Use BoltzGen's output_dir if available, otherwise fall back to output or default
        if 'output_dir' in config:
            output_dir = config['output_dir']
        else:
            output_dir = config.get("output", "anyfold_folded")

        # Make output_dir absolute path for AnyFold
        if not os.path.isabs(output_dir):
            output_dir = os.path.abspath(output_dir)

        print(f"AnyFold output directory: {output_dir}")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load designed sequences and targets
        designed_structures = self._load_boltzgen_structures(input_dir)
        target_sequences = self._load_target_sequences()

        # Create JSON files for AnyFold
        temp_dir = tempfile.mkdtemp(prefix="boltzgen_anyfold_")

        try:
            # Create cofolding JSON files
            for structure_data in designed_structures:
                structure_id = structure_data.get('id', 'structure')
                target_id = structure_data.get('target_id', '')

                # Create AnyFold JSON structure
                anyfold_structure = {
                    "A": {
                        "type": "protein",
                        "sequence": target_sequences.get(target_id, "")
                    },
                    "B": {
                        "type": "protein",
                        "sequence": structure_data['sequence']
                    }
                }

                json_path = os.path.join(temp_dir, f"{structure_id}_complex.json")
                with open(json_path, 'w') as f:
                    json.dump(anyfold_structure, f, indent=2)

                print(f"Created: {structure_id}_complex.json")

            # Create AnyFold config (matching cofolding.yaml structure)
            # AnyFold should automatically pick up TINYPROT_CACHE, so leave msa_dir empty
            anyfold_config = OmegaConf.create({
                "input": temp_dir,
                "msa_dir": "",  # Let AnyFold/tinyprot handle MSA path automatically
                "output": output_dir,  # Already made absolute above
                "recycling_steps": 4,
                "diffusion_samples": 1,
                "diffusion_steps": 200,  # Match anybind
                "num_seeds": 1,
                "skip_existing": True,
                "assert_msa": False,  # Keep false to make MSAs optional
                "save_traj": False,
                "save_distogram": False,
                "save_full_confidence": False,
                "diffusion": {
                    "sigma_min": 0.0004,
                    "sigma_max": 160.0,
                    "sigma_data": 16.0,
                    "edm_churn": True,
                    "rho": 7,
                    "gamma_0": 0.8,
                    "gamma_min": 1.0,
                    "noise_scale": 1.003,
                    "step_scale": 1.0,  # Match anybind
                }
            })

            # Run AnyFold directly (following anyfold/__main__.py pattern)
            print("Running AnyFold cofolding...")

            # Load model config
            import pytorch_lightning as pl
            from anyfold.model.model import AnyFoldModel
            from anyfold.utils.load_weights import load_weights
            from anyfold.data.utils import collate
            from torch.utils.data import DataLoader

            model_config_path = "/data/cb/mihirb14/projects/anyfold_dev/anyfold/model/config.yaml"
            model_cfg = OmegaConf.load(model_config_path)

            # Create model and load weights
            model = AnyFoldModel(model_cfg)
            load_weights("/data/cb/scratch/share/anyfold_contact.ckpt", model)

            # Create task and run
            task = CofoldingTask(anyfold_config)
            model.inference_task = task

            # Create trainer and run
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                logger=False,
                enable_progress_bar=True
            )

            loader = DataLoader(task, batch_size=1, collate_fn=collate, num_workers=1)
            trainer.predict(model, loader)

            print("AnyFold completed!")

        finally:
            # Cleanup temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _load_boltzgen_structures(self, input_dir: str):
        """Load BoltzGen designed structures from CIF files"""
        structures = []

        # Look for CIF files in the input directory
        import glob
        cif_files = glob.glob(os.path.join(input_dir, "*.cif"))

        for cif_file in cif_files:
            structure_id = os.path.splitext(os.path.basename(cif_file))[0]
            sequences = self._extract_sequences_from_cif(cif_file)

            if sequences:
                # The designed binder should be chain B (entity 1), target is chain A (entity 2)
                binder_sequence = sequences.get('1', '')  # Entity 1 = designed binder
                if binder_sequence:
                    structures.append({
                        'id': structure_id,
                        'target_id': structure_id,  # Use the same ID for target lookup
                        'sequence': binder_sequence
                    })

        print(f"Loaded {len(structures)} structures from CIF files")
        return structures

    def _load_target_sequences(self):
        """Load target protein sequences"""
        target_sequences = {}

        # Look for PDB files in targets directories
        for targets_dir in [Path("targets"), Path("../targets")]:
            if targets_dir.exists():
                for pdb_file in targets_dir.glob("*.pdb"):
                    target_name = pdb_file.stem
                    sequence = self._extract_sequence_from_pdb(pdb_file)
                    if sequence:
                        target_sequences[target_name] = sequence

        print(f"Loaded target sequences: {list(target_sequences.keys())}")
        return target_sequences

    def _extract_sequence_from_pdb(self, pdb_path):
        """Extract amino acid sequence from PDB file"""
        aa_3to1 = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }

        sequence = ""
        prev_res_num = None

        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                        res_name = line[17:20].strip()
                        res_num = int(line[22:26].strip())

                        if res_num != prev_res_num:
                            if res_name in aa_3to1:
                                sequence += aa_3to1[res_name]
                            prev_res_num = res_num
        except Exception as e:
            print(f"Warning: Could not extract sequence from {pdb_path}: {e}")
            return ""

        return sequence

    def _extract_sequences_from_cif(self, cif_path):
        """Extract sequences from CIF file"""
        sequences = {}

        try:
            with open(cif_path, 'r') as f:
                content = f.read()

            # Parse the entity_poly section for sequences
            lines = content.split('\n')
            in_entity_poly_loop = False

            for i, line in enumerate(lines):
                line = line.strip()

                # Look for the start of entity_poly loop
                if line == 'loop_' and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line == '_entity_poly.entity_id':
                        in_entity_poly_loop = True
                        continue

                # Process data lines in the entity_poly loop
                if in_entity_poly_loop:
                    if line.startswith('_entity_poly.'):
                        continue  # Skip header lines
                    elif line.startswith('loop_') or line.startswith('#') or not line:
                        in_entity_poly_loop = False  # End of this loop
                        break
                    else:
                        # This is a data line: entity_id type strand_id sequence
                        parts = line.split()
                        if len(parts) >= 4:
                            entity_id = parts[0]
                            sequence = parts[3]  # pdbx_seq_one_letter_code
                            sequences[entity_id] = sequence

        except Exception as e:
            print(f"Warning: Could not extract sequences from {cif_path}: {e}")

        return sequences