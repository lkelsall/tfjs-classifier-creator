const { readFileSync, readdirSync } = require('fs');
const tf = require('@tensorflow/tfjs-node');

exports.tensorFromJpg = (filePath) => {
  const imageData = readFileSync(filePath);

  const imgTensor = tf.node.decodeImage(new Uint8Array(imageData), 3);

  return imgTensor;
};

exports.trainClassFromImages = (classifier, filePath, model, className) => {
  const imageFileNames = readdirSync(filePath);

  imageFileNames.forEach((imageFile) => {
    const imageTensor = this.tensorFromJpg(`${filePath}/${imageFile}`);
    const logits = model.infer(imageTensor);
    classifier.addExample(logits, className);
  });
};
