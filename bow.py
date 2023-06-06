import glob
import random
random.seed(123123)
    
import cv2
import numpy as np
import dbow

from tqdm import tqdm

# Load Images
images_path = glob.glob('./CUSTOM_sequence/images/*.jpg')
print(images_path)
images = []
for i, image_path in enumerate(images_path):
    if i % 10 == 0:
        images.append(cv2.imread(image_path))

print(f"Num images: {len(images)}")



# Create Vocabulary
n_clusters = 150
depth = 5
vocabulary = dbow.Vocabulary(images, n_clusters, depth)

orb = cv2.ORB_create()

# Convert images to Bag of Binary Words and calculate scores between them
bows = []
for image in tqdm(images):
    kps, descs = orb.detectAndCompute(image, None)
    descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
    bows.append(vocabulary.descs_to_bow(descs))

for i in range(len(bows)):
    for j in range(len(bows)):
        print(f'Similarity between Image {i} and Image {j} = {bows[i].score(bows[j])}')
    print('\n')


# Create a database
db = dbow.Database(vocabulary)
for image in images:
    kps, descs = orb.detectAndCompute(image, None)
    descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
    db.add(descs)