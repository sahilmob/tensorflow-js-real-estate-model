import * as tf from "@tensorflow/tfjs";

import { TRAINING_DATA } from "./real-estate-data";
import "./App.css";

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUT_TENSOR = tf.tensor2d(INPUTS);
const OUTPUT_TENSOR = tf.tensor1d(OUTPUTS);

function normalize(tensor: tf.Tensor2D, min?: tf.Tensor, max?: tf.Tensor) {
  const result = tf.tidy(() => {
    const MIN_VALUES = min || tf.min(tensor, 0);
    const MAX_VALUES = max || tf.max(tensor, 0);

    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return {
      NORMALIZED_VALUES,
      MIN_VALUES,
      MAX_VALUES,
    };
  });

  return result;
}

const FEATURE_RESULT = normalize(INPUT_TENSOR);

FEATURE_RESULT.NORMALIZED_VALUES.print();
FEATURE_RESULT.MIN_VALUES.print();
FEATURE_RESULT.MAX_VALUES.print();

INPUT_TENSOR.dispose();

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [2] }));
model.summary();

async function train() {
  const LEARNING_RATE = 0.01;

  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: tf.losses.meanSquaredError,
  });

  const result = await model.fit(
    FEATURE_RESULT.NORMALIZED_VALUES,
    OUTPUT_TENSOR,
    {
      validationSplit: 0.15,
      shuffle: true,
      batchSize: 64,
      epochs: 10,
    }
  );

  OUTPUT_TENSOR.dispose();
  FEATURE_RESULT.NORMALIZED_VALUES.dispose();

  console.log(
    "Average error loss:" +
      Math.sqrt(result.history.loss[result.history.loss.length - 1] as number)
  );
  console.log(
    "Average validation loss:" +
      Math.sqrt(
        result.history.val_loss[result.history.val_loss.length - 1] as number
      )
  );

  evaluate();
}

train();

function evaluate() {
  tf.tidy(() => {
    const newInput = normalize(
      tf.tensor2d([[750, 1]]),
      FEATURE_RESULT.MIN_VALUES,
      FEATURE_RESULT.MAX_VALUES
    );
    const output = model.predict(newInput.NORMALIZED_VALUES) as tf.Tensor;
    output.print();
  });

  FEATURE_RESULT.MIN_VALUES.dispose();
  FEATURE_RESULT.MAX_VALUES.dispose();
  model.dispose();

  console.log(tf.memory().numTensors);
}

function App() {
  return <></>;
}

export default App;
