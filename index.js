const mobilenetModule = require('@tensorflow-models/mobilenet');
const knnClassifier = require('@tensorflow-models/knn-classifier');
const { trainClassFromImages, tensorFromJpg } = require('./utils');
const { dataSync } = require('@tensorflow/tfjs-node');
const fs = require('fs');

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

  const datasetTensor = classifier.getClassifierDataset();
  const datasetObject = {};
  Object.keys(datasetTensor).forEach((key) => {
    const data = datasetTensor[key].dataSync();

    datasetObject[key] = Array.from(data);
  });
  const jsonStr = JSON.stringify(datasetObject);

  fs.writeFileSync('./datasetString.txt', jsonStr);

  return jsonStr;
};

createClassifierData(dogNames);
