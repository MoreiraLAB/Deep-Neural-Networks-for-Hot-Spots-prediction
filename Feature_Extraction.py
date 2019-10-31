import os
import Bio
import pandas as pd
from Bio.PDB import *

class utilities:

    def __init__(self):

        self.amino_acids = ['CYS', 'ASP', 'SER', 'GLN', 'LYS',
                            'ILE', 'PRO', 'THR', 'PHE', 'ASN', 
                            'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 
                            'ALA', 'VAL', 'GLU', 'TYR', 'MET']
        self.converter = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
                 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
                 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    def table_opener(self, file_path, sep = ","):

        opened_decoder = pd.read_csv(file_path, sep = sep, header = 0)
        return opened_decoder

class features:

    def __init__(self, row):
        self.row = row

    def retrieve_sequence(self, input_sequences):
        self.complex_name = self.row["CPX"].lower()
        self.chain = self.row["PDBChain"]
        self.res_number = self.row["PDBResNo"]
        chain_sequence = input_sequences[self.complex_name][self.chain]
        encoded_sequence = generate_encoded(chain_sequence)
        return encoded_sequence

    def location_features(self, input_sequences, target_residue):

        self.sequence_table = self.retrieve_sequence(input_sequences)
        order_list = list(range(0,self.sequence_table.shape[0]))
        ordering = pd.DataFrame(order_list, columns = ["order"])
        inverse_ordering = pd.DataFrame(order_list[::-1], columns = ["reverse_order"])
        pseudo_distance = pd.DataFrame(list(range(0,target_residue - self.sequence_table[class_id_output].iloc[0] + 1))[::-1] 
                                        + list(range(1, self.sequence_table[class_id_output].iloc[-1] - target_residue + 1)), columns = ["pseudo_distance"])
        return pd.concat([self.sequence_table, ordering / ordering.max(), inverse_ordering / inverse_ordering.max(), pseudo_distance / pseudo_distance.max()], axis = 1)

def get_unique(input_df):

    from Bio.PDB import PDBList
    unique_pdbs = input_df.CPX.unique()
    pdbl = PDBList()
    for single_pdb in unique_pdbs:
        pdbl.retrieve_pdb_file(single_pdb, pdir='PDB', file_format = "pdb")

def retrieve_sequence_raw(input_folder = "PDB", system_sep = "/"):
   
    target_folder = os.getcwd() + system_sep + input_folder
    output_dict = {}
    for files in os.listdir(target_folder):
        parser = PDBParser()
        target_file = os.getcwd() + system_sep + input_folder + system_sep + files
        structure = parser.get_structure(files[0:-4], target_file)
        pdb_id, pdb_dict = structure.id[3:], {}
        for model in structure:
            for chain in model:
                chain_dict, chain_name, sequence = {}, chain.id, ""
                for residue in chain:
                    res_number, res_name = residue.get_full_id()[-1][1], residue.resname
                    if res_name in utilities().amino_acids:
                        single_letter = utilities().converter[res_name]
                        sequence += single_letter
                        chain_dict[res_number] = res_name
                pdb_dict[chain_name] = chain_dict
        output_dict[pdb_id] = pdb_dict
    return output_dict

def generate_encoded(input_sequence):
    output_table = []
    encoded_table = utilities().table_opener(encoder_path)
    for residue_number in input_sequence.keys():
        residue_letter = utilities().converter[input_sequence[residue_number]]       
        encoded_residue = encoded_table.loc[encoded_table[class_id_name] == residue_letter].iloc[:,1:]
        proper_row = [residue_number] + list(encoded_residue.values[0])
        output_table.append(proper_row)
    header = [class_id_output] + list(encoded_residue)
    return pd.DataFrame(output_table, columns = header)

def generate_file(input_file, residues_features):

    prepared_table, classes = [], []

    for row in range(input_file.shape[0]):        
        current_row = input_file.iloc[row]    
        current_properties = pd.DataFrame(features(current_row).location_features(residues_features, current_row[class_id_original]))
        writeable_row = list(current_row.values) + \
                            current_properties.loc[current_properties[class_id_output] == current_row[class_id_original]].values.tolist()[0]
        if current_properties.isnull().any().any() == True: continue
        prepared_table.append(writeable_row)
        if current_row[class_name] == NS: classe = 0
        elif current_row[class_name] == HS: classe = 1
        classes.append(classe)

    return pd.DataFrame(prepared_table), pd.DataFrame(classes, columns = [class_name])

encoder_path = os.getcwd() + "/resources/encoding.csv"
output_features_name = "spoton_clean.csv"
output_class_name = "class_clean.csv"
class_id_original = "PDBResNo"
class_id_output = "res_number"
class_id_name = "res_letter"
class_name = "Classe"
NS, HS = "NS", "HS"


opened_file = utilities().table_opener("spoton.csv")
#get_unique(opened_file)

residues_dict = retrieve_sequence_raw()
novel_features, classes = generate_file(opened_file, residues_dict)
novel_features.to_csv(output_features_name, sep = ",", index = False)
classes.to_csv(output_class_name, sep = ",", index = False)
