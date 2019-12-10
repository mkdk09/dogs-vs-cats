import os, shutil

original_dataset_dir = 'dogs-vs-cats-data/train'

base_dir = 'cats_and_dogs_small'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

filenames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for filename in filenames:
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(train_cats_dir, filename)
    shutil.copyfile(src, dst)

filenames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for filename in filenames:
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(validation_cats_dir, filename)
    shutil.copyfile(src, dst)

filenames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for filename in filenames:
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(test_cats_dir, filename)
    shutil.copyfile(src, dst)


filenames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for filename in filenames:
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(train_dogs_dir, filename)
    shutil.copyfile(src, dst)

filenames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for filename in filenames:
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(validation_dogs_dir, filename)
    shutil.copyfile(src, dst)

filenames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for filename in filenames:
    src = os.path.join(original_dataset_dir, filename)
    dst = os.path.join(test_dogs_dir, filename)
    shutil.copyfile(src, dst)