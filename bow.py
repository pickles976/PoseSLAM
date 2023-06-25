from lib.pydbow import BagOfWords
import os

dataset = "08"
bow = BagOfWords(n_clusters=150)
bow.generate_from_images(os.path.join(dataset,"images"))
bow.save(os.path.join(dataset, "dictionary"))