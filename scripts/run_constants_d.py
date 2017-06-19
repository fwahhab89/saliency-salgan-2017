# Work space directory
HOME_DIR = '/home/faiz/Desktop/saliency-experiments/salgan'

# Path to SALICON raw data
pathToImages = '/home/faiz/Desktop/datasets/images'
pathToMaps = '/home/faiz/Desktop/datasets/saliency'
pathToFixationMaps = '/home/faiz/Desktop/datasets/fixation'

# Path to processed data
pathOutputImages = '/home/faiz/Desktop/datasets/processed-data/images320x240'
pathOutputMaps = '/home/faiz/Desktop/datasets/processed-data/saliency320x240'
#pathToPickle = '/home/faiz/Desktop/datasets/processed-data/pickle320x240'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/home/faiz/Desktop/datasets/processed-data/training_images.txt'
VAL_DATA_DIR = '/home/faiz/Desktop/datasets/processed-data/validation_images.txt'
#TEST_DATA_DIR = '/home/faiz/Desktop/datasets/processed-data/pickle256x192/testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/home/faiz/Desktop/saliency-experiments/salgan/vgg16.pkl'

# Input image and saliency map size
INPUT_SIZE = (320, 240)

# Directory to keep snapshots
DIR_TO_SAVE_BCE = '/home/faiz/Desktop/results/bce_vanilla_320x240d'
DIR_TO_SAVE_SALGAN = '/home/faiz/Desktop/results/salgan_vanilla_320x240'
