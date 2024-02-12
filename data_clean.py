import pandas as pd
import os
from PIL import Image

parent_folder = os.getcwd()
input_directory = parent_folder + "\\Data\\no_graffiti\\"
output_directory = parent_folder + "\\test\\"

# Change the directory 
os.chdir(input_directory) 

count = 0
#print(os.listdir())
for file in os.listdir(): 
    try:
        im = Image.open(file)
        im.save(output_directory + file[:-4] + str(count) + ".png")
        count += 1
        print(count)
    except:
        print("File format invalid")
print("Done")