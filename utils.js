const { readFileSync } = require('fs');
const tf = require('@tensorflow/tfjs-node');

const tensorFromJpg = (filePath) => {
  const imageData = readFileSync(filePath);

  const imgTensor = tf.node.decodeImage(new Uint8Array(imageData), 3);

  return imgTensor;
};
