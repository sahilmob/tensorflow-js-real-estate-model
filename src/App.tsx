import * as tf from "@tensorflow/tfjs";

import "./App.css";

const INPUTS = [];
const OUTPUTS = [];
const LEARNING_RATE = 0.0001;
const OPTIMIZER = tf.train.sgd(LEARNING_RATE);

for (let i = 0; i <= 20; i++) {
  INPUTS.push(i);
}

for (let i = 0; i <= 20; i++) {
  OUTPUTS.push(INPUTS[i] * INPUTS[i]);
}

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUT_TENSOR = tf.tensor1d(INPUTS);
const OUTPUT_TENSOR = tf.tensor1d(OUTPUTS);

function normalize(tensor: tf.Tensor1D, min?: tf.Tensor, max?: tf.Tensor) {
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
model.add(tf.layers.dense({ units: 25, inputShape: [1], activation: "relu" }));
model.add(tf.layers.dense({ units: 5, activation: "relu" }));
model.add(tf.layers.dense({ units: 1 }));
model.summary();

function logProgress(epoch: number, logs: tf.Logs | undefined) {
  console.log(
    "Epoch: " + epoch + " Loss: " + logs?.loss ? Math.sqrt(logs!.loss) : 0
  );

  if (epoch === 70) {
    OPTIMIZER.setLearningRate(LEARNING_RATE / 2);
  }
}

async function train() {
  model.compile({
    optimizer: OPTIMIZER,
    loss: tf.losses.meanSquaredError,
  });

  const result = await model.fit(
    FEATURE_RESULT.NORMALIZED_VALUES,
    OUTPUT_TENSOR,
    {
      callbacks: { onEpochEnd: logProgress },
      shuffle: true,
      batchSize: 2,
      epochs: 200,
    }
  );

  OUTPUT_TENSOR.dispose();
  FEATURE_RESULT.NORMALIZED_VALUES.dispose();

  console.log(
    "Average error loss:" +
      Math.sqrt(result.history.loss[result.history.loss.length - 1] as number)
  );

  evaluate();
}

train();

function evaluate() {
  tf.tidy(() => {
    const newInput = normalize(
      tf.tensor1d([7]),
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
