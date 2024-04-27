import * as tf from "@tensorflow/tfjs";

import { TRAINING_DATA } from "./real-estate-data";
import "./App.css";

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

function App() {
  return <></>;
}

export default App;
