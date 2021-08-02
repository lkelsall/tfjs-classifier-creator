const mobilenetModule = require('@tensorflow-models/mobilenet');
const knnClassifier = require('@tensorflow-models/knn-classifier');
const { trainClassFromImages, tensorFromJpg } = require('./utils');

const dogNames = [
  'Labrador Retriever',
  'Cocker Spaniel',
  'Staffordshire Bull Terrier',
  'French Bulldog',
  'Border Collie',
  'Shih Tzu',
  'Chihuahua',
  'German Shepherd',
  'Golden Retriever',
  'Yorkshire Terrier',
  'English Springer Spaniel',
  'Pug',
  'Beagle',
  'West Highland White Terrier',
  'Pomeranian',
  'Rottweiler',
  'Poodle',
  'Boxer',
];

const createClassifierData = async (dogNames) => {
  const classifier = knnClassifier.create();
  const model = await mobilenetModule.load();

  dogNames.forEach((dogName) => {
    trainClassFromImages(
      classifier,
      `../dog-images/${dogName}`,
      model,
      dogName
    );
  });

  const tensor = tensorFromJpg('./beagleimg.jpg');

  const logits = model.infer(tensor);

  classifier.predictClass(logits).then((result) => {
    console.log(result);
  });
};

createClassifierData(dogNames);
