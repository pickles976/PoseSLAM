from lib.pydbow import BagOfWords

bow = BagOfWords(n_clusters=150)
bow.generate_from_images("./00/images")
bow.save("./00/dictionary")