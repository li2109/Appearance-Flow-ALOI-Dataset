import os
from shutil import copyfile

dataset_dir = "Selected"

if not os.path.exists(dataset_dir):                                                                                                                                           
	os.makedirs(dataset_dir)

for folder in range(900):
	folder_number = str(folder+1)
	path = os.path.join(dataset_dir, folder_number)
	if not os.path.exists(path):
		os.makedirs(path)

source_dir = "masked_256"
pic_selected = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

for folder in range(900): 
	folder_number = str(folder+1)
	for index in range(12):
		pic_num = str(folder_number)+"_r"+str(pic_selected[index])+".png"
		des_path = os.path.join(dataset_dir, folder_number, pic_num)
		source_path = os.path.join(source_dir, folder_number, pic_num)
		copyfile(source_path, des_path)