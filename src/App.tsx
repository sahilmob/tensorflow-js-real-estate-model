import * as tf from "@tensorflow/tfjs";

import { TRAINING_DATA } from "./real-estate-data";
import "./App.css";

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUT_TENSOR = tf.tensor2d(INPUTS);
const OUTPUT_TENSOR = tf.tensor1d(OUTPUTS);

function normalize(tensor: tf.Tensor2D, min?: tf.Tensor1D, max?: tf.Tensor1D) {
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

function App() {
  return <></>;
}

export default App;
