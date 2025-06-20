#!/usr/bin/env python
# coding: utf-8

# # FEgrow: An Open-Source Molecular Builder and Free Energy Preparation Workflow
# 
# **Authors: Mateusz K Bieniek, Ben Cree, Rachael Pirie, Joshua T. Horton, Natalie J. Tatum, Daniel J. Cole**

# ## Overview
# 
# Building and scoring molecules can be further streamlined by employing our established protocol. Here we show how to quickly build a library and score the entire library. 

# In[1]:

import os   # afk
from glob import glob  #afk 

import pandas as pd
import prody
from rdkit import Chem

import fegrow
from fegrow import ChemSpace

from fegrow.testing import core_5R83_path, rec_5R83_path, data_5R83_path

from dask.distributed import LocalCluster

def main():

    OUTPUT_DIR = "fegrow_result"
    
     
    lc = LocalCluster(processes=True, n_workers=None, threads_per_worker=1)
    
    counter = 1
    # In[14]:

    input_folder = "./apodockRec-H"
    # Find all .pdb files in the input folder
    pdb_files = glob(os.path.join(input_folder, "*.pdb"))

    for pdb_file in pdb_files:
    # # Prepare the ligand template
        print(f" pdb file {counter} read in")
        # In[2]:


        #!grep "XEY" 7l10.pdb > in.pdb
        #!obabel -ipdb lig-rebuilt.pdb -O in-H.sdf -p 7


        # In[3]:


        #scaffold = Chem.SDMolSupplier(core_5R83_path)[0]
        scaffold = Chem.SDMolSupplier('coreh.sdf')[0]


        # In[4]:


        #toview = fegrow.RMol(scaffold)
        #toview.rep2D(idx=True, size=(500, 500))


        # In[5]:


        with open('smiles-released-mers-mol0.txt') as f:
         
            mols = f.read().splitlines()


        # In[6]:


        #mols[0]


        # In[7]:


        #mols = ['C1(OCCC)=CC=CN=C1',
        #        'CNC(=O)CN1C[C@@]2(C(=O)N(c3cncc4ccccc34)C[C@@H]2CNc2ccncn2)c2cc(Cl)ccc2C1=O',
        #        'CNC(=O)CN1C[C@@]2(C(=O)N(c3cncc4ccccc34)C[C@@H]2CNc2cnn(C)c2)c2cc(Cl)ccc2C1=O',
        #        'Cc1cnc(CN2C[C@@]3(C(=O)N(c4cncc5ccccc45)C[C@@H]3C)c3cc(F)ccc3C2=O)cn1',
        #        'CNC(=O)CN1C[C@@]2(C(=O)N(c3cncc4ccccc34)C[C@@H]2COC(C)C)c2cc(Cl)ccc2C1=O',
        #        'C[C@H]1CN(c2cncc3ccccc23)C(=O)[C@@]12CN(Cc1nccn1C)C(=O)c1ccc(F)cc12'
        #       ]


        # In[8]:


        #Chem.MolFromSmiles(mols[0])


        # In[ ]:





        # In[9]:


        #pattern = scaffold

        #for i in range(len(mols)):
        #    mol = Chem.MolFromSmiles(mols[i])
        #    if mol.HasSubstructMatch(pattern) == False:
        #        print(i, mols[i])


        # In[ ]:





        # As we are using already prepared Smiles that have the scaffold as a substructure, it is not needed to set any growing vector. 

        # In[ ]:





        # In[10]:


       


        # In[ ]:



        print(f"loading core finished round {counter}")
        print(f"creating chemspace with dask round {counter}")

        # In[11]:


        # create the chemical space
        cs = ChemSpace(dask_cluster=lc)   



        # In[12]:


        #cs._dask_cluster


        # In[ ]:





        # In[13]:


        # we're not growing the scaffold, we're superimposing bigger molecules on it
        cs.add_scaffold(scaffold)


        # In[15]:


        # load 50k Smiles
        #smiles = pd.read_csv('csv/arthor-hits-2024Mar26-0918.csv',
        #                     names=["Smiles", "??", "db"],
        #                     index_col=0).Smiles

        #smiles = pd.read_csv('smiles.csv').Smiles.to_list()


        # take all 20000
        #smiles = smiles.apply(lambda r: r.split()[0])
        smiles = mols[0:]

        # here we add Smiles which should already have been matched
        # to the scaffold (rdkit Mol.HasSubstructureMatch)
        #cs.add_smiles(smiles.to_list(), protonate=True)
        cs.add_smiles(smiles, protonate=True)
        cs

        # get the protein-ligand complex structure
        #!wget -nc https://files.rcsb.org/download/3vf6.pdb

        

        # load the complex with the ligand
        sys = prody.parsePDB(pdb_file)

        # remove any unwanted molecules
        rec = sys.select('not (nucleic or hetatm or water)')

        # save the processed protein
        prody.writePDB('rec.pdb', rec)

        # fix the receptor file (missing residues, protonation, etc)
        os.mkdir(OUTPUT_DIR)
        fegrow.fix_receptor("rec.pdb", f"{OUTPUT_DIR}/rec_final_{counter}.pdb")
        print(f"pdb file into rec_final {counter}")
        # load back into prody
        #rec_final = prody.parsePDB("rec_final.pdb")
        #rec_final = prody.parsePDB("out.pdb")

        # fix the receptor file (missing residues, protonation, etc)
        ##fegrow.fix_receptor("7t79-H-prep.pdb", "rec_final.pdb")

        # load back into prody
        ##rec_final = prody.parsePDB("rec_final.pdb")

        #!grep "ATOM" ../structures/7t79-H.pdb > rec_final.pdb
        #cs.add_protein(rec_5R83_path)
        cs.add_protein(f"rec_final_{counter}.pdb")
        print(f"successfully added pdb {counter} to chemspace to evaluate conformers on it")


        # In[16]:


        cs.evaluate(num_conf=500, gnina_gpu=False, penalty=0.0, al_ignore_penalty=False)


        # In[ ]:





        # In[17]:


        #cs.df afk


        # In[18]:

        
        cs.to_sdf(f"{OUTPUT_DIR}/cs_optimised_molecules_in_rec_{counter}.sdf")


        # In[ ]:





        # In[19]:


        for i in range (len(cs)):
            try:
                cs[i].to_file("best_conformers_in_rec_{0}_{1}.pdb".format(counter,i)) # afk
            except AttributeError:
                print("No conformer for molecule", i)


        # In[ ]:





        # In[20]:


        cs.df.to_csv('MERS-out.csv', index=True)
      
        counter += 1
     

        # cs.clear()
        # afk need to reset/clear cs for next iteration


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Especially needed for frozen executables
    main()
