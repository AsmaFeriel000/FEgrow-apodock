from chimera import runCommand as rc

rc("open complex-mers-lig-h.sdf_pack_5-with-core.pdb")
rc("addh hbond true")
rc("write format pdb 0 output_with_h.pdb")
rc("stop now")
