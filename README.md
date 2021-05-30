Sample code for debugging high CPU memory

Download the feats file in hdf5 format "https://drive.google.com/file/d/1vEer5fhbTLA-x8HfR_1FhhmJKLSLsv3w/view?usp=sharing" and copy to feats/val/
(In the script main.py I am loading data one at a time because I have very large data that cant be loaded into memort at once.)

Run the commnad "python3 main.py --train-data-dir data/val --val-data-dir data/val"
(Just for debugging purpose I kept the train and val set to be same.)# asv-sample