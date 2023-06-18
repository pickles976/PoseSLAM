from lib.pydbow import BagOfWords

bow = BagOfWords(n_clusters=150)
bow.generate_from_images("./03/images")
bow.save("./03/dictionary")