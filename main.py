import re   
import os
import glob 
import datetime, time
import json

# ------Preprocess-------
scrape_dir = 'data'
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H%M%S')

print("Converting sequences ... ")
out_file = os.path.join('data', 'protein-seqs-' + st + '.txt')
print("Writing to: %s" % out_file)

num_proteins_done = 0   # TODO: Remove (here to reduce complexity)
fasta_files = glob.glob(scrape_dir + "/*.fasta") 
print(fasta_files)

def dump_to_file(protein_id, sequence):
    with open(out_file, "a") as f:
        f.write(protein_id + "," + sequence + "\n")

for fname in fasta_files:
    print("Converting: %s: " % fname)
    proteins = {}   # will hold all proteins in this form ->  id: seq

    with open (fname, 'r') as f:
        protein_seq = ''
        protein_id = ''
        
        for line in f:
            
            # Match this:   >[two chars]|[alphanumeric chars]|   
            
            match = re.search(r'^>([a-z]{2})\|([A-Z0-9]*)\|', line) 
            if match:
                if protein_id != '': 
                    dump_to_file(protein_id, protein_seq)
                
                num_proteins_done += 1 
                if num_proteins_done > 1000: break   # TODO: Remove                     
                    
                protein_id = match.group(2)
                protein_seq = ''   
    
            else:
                protein_seq += line.strip()
                
        if protein_id != '':
            dump_to_file(protein_id, protein_seq)

# convert function
print("Converting functions ...") 
out_file_fns = os.path.join('data', 'protein-functions-' + st + '.txt')
print(out_file_fns)
target_functions = ['0005524']   # just ATP binding for now 
annot_files = glob.glob(scrape_dir + "/*annotations.txt")
print(annot_files)

has_function = []  # a dictionary of protein_id: boolean  (which says if the protein_id has our target function)

for fname in annot_files:
    with open (fname, 'r') as f:
        for line in f:
            match = re.search(r'([A-Z0-9]*)\sGO:(.*);\sF:.*;', line)
            if match:
                protein_id = match.group(1)
                function = match.group(2)
                
                if function not in target_functions:
                        continue
                        
                has_function.append(protein_id) 
          
    
    with open(out_file_fns, 'w') as fp:
        json.dump(has_function, fp)
        
    # Take a peek 
    print(has_function[:10])

# -----Dataset Manipulation-----
import numpy as np
np.random.seed(316)
from sklearn.model_selection import train_test_split

sequences_file = os.path.join('data', 'protein-seqs-'+ st +'.txt') 
functions_file = os.path.join('data', 'protein-functions-' + st + '.txt')

with open(functions_file) as fn_file:
    has_function = json.load(fn_file)

max_seq_length = 500 # Sequences length varies. look at the data for min value
x = []
y = []
pos_examples = 0 
neg_examples = 0

with open(sequences_file) as f:
    for line in f:
        ln = line.split(',')
        protein_id = ln[0].strip()
        seq = ln[1].strip()

        if len(seq) > max_seq_length:
            continue     

        else:
            seq += (max_seq_length-len(seq))*'_'   

        x.append(seq)

        if protein_id in has_function:
            y.append(1)
            pos_examples+=1

        else: 
            y.append(0)
            neg_examples+=1

print("Number of positive examples: ", pos_examples)
print("Number of negative examples: ", neg_examples)

def sequence_to_indices(sequence):
    try:
        acid_letters = ['_', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
        indices = [acid_letters.index(c) for c in list(sequence)]
        return indices
    
    except Exception:
        print(sequence)
        raise Exception
        
x_final = []
for i in range(len(x)):
    temp_x = sequence_to_indices(x[i])
    x_final.append(temp_x)

x_final = np.array(x_final)
y_final = np.array(y)

# -----Training-----
from tensorflow.keras.models import Model, Sequential
from keras.layers import Embedding, Input, Flatten, Dense, Activation
from keras.optimizers import SGD

n = x_final.shape[0]
randomize = np.arange(n)
np.random.shuffle(randomize)

x_final = x_final[randomize]
y_final = y_final[randomize]
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.3)

num_amino_acids = 23
embedding_dims = 10
nb_epoch = 60
batch_size = 128

model = Sequential()
model.add(Embedding(num_amino_acids, embedding_dims, input_length=max_seq_length))
model.add(Flatten())
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, 
                    batch_size = batch_size, epochs = nb_epoch,
                    validation_data = (x_test, y_test),
                    verbose = 1)

loss, accuracy = model.evaluate(x_test, y_test)
print("loss: ", loss, "\naccuracy: ", accuracy)


