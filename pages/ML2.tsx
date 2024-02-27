import { Fragment, useState } from "react";

import Head from "next/head";

import * as tf from "@tensorflow/tfjs";

export default function ML2() {
  const [predictedValue, setPredcitedValue] = useState<null | number>(null);
  const [message, setMessage] = useState("");

  async function loadModelLocally() {
    const model = await tf.loadLayersModel("localstorage://my-model");
    //console.log(model.summary(), model.weights);

    // Make a prediction
    const predict = model.predict(tf.tensor2d([5], [1, 1])) as tf.Tensor;
    //console.log(predict.dataSync()[0]);
    setPredcitedValue(predict.dataSync()[0]);
  }

  async function loadModelPublic() {
    // Construct the full path to the model file inside the public folder
    //const modelPath = "http://localhost:3000/my-model.json";
    const modelPath = "https://crs-se-mfa-support.vercel.app/my-model.json";

    // Load the model using the constructed path
    const model = await tf.loadLayersModel(modelPath);

    // Make a prediction
    const predict = model.predict(tf.tensor2d([5], [1, 1])) as tf.Tensor;
    //console.log(predict.dataSync()[0]);
    setPredcitedValue(predict.dataSync()[0]);
  }

  async function regressionTest() {
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

    // Specify some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

    // Fit the model
    model.fit(xs, ys, { epochs: 400 });

    // Make a prediction
    const predict = model.predict(tf.tensor2d([5], [1, 1])) as tf.Tensor;
    setPredcitedValue(predict.dataSync()[0]);

    // Save the trained model
    await model.save("localstorage://my-model");
    // await model.save("downloads://my-model");

    //console.log(model.summary(), model.weights);
    //console.log(predict.dataSync()[0]);
  }

  return (
    <Fragment>
      <Head>
        <title>About / Try Dear!</title>
      </Head>
      <div>{message}</div>
      <h1>Machine Learning</h1>
      <button onClick={regressionTest}>Regression test</button>
      <button onClick={loadModelLocally}>Load model locally</button>
      <button onClick={loadModelPublic}>Load model public</button>
      {predictedValue}
    </Fragment>
  );
}
