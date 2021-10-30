async function runModel(event) {
  event.preventDefault();

  const input = tf.input({shape: [200,]});

  const embeddingsLayer = tf.layers.embedding({ inputDim: 20000, outputDim: 128, inputLength: 200, trainable: true });
  const embbeded_sequences = embeddingsLayer.apply(input)
  const spatialDropout1dLayer = tf.layers.spatialDropout1d({rate: 0.35});
  const spatialDropout1dLayer_out = spatialDropout1dLayer.apply(embbeded_sequences);
  const lstm = tf.layers.bidirectional({layer: tf.layers.lstm({units: 60, returnSequences: true, dropout: 0.15, recurrentDropout: 0.15 })}); // should be bidirectional? what about dropout?
  const lstm_out = lstm.apply(spatialDropout1dLayer_out);

  const x_a1 = tf.layers.dense({units: 120, kernelInitializer: 'glorotUniform', activation: 'tanh'});
  const x_a1_out = x_a1.apply(lstm_out)

  const x_a2 = tf.layers.dense({units: 1, kernelInitializer: 'glorotUniform', activation: 'linear'});
  const x_a2_out = x_a2.apply(x_a1_out)

  const flattenLayer = tf.layers.flatten();
  const flattenLayer_out = flattenLayer.apply(x_a2_out);

  const act = tf.layers.activation({activation: 'softmax'});
  const act_out = act.apply(flattenLayer_out)

  const repeatLayer = tf.layers.repeatVector({n: 120});
  const repeatLayer_out = repeatLayer.apply(act_out)

  const permuteLayer = tf.layers.permute({dims: [2,1]});
  const permuteLayer_out = permuteLayer.apply(repeatLayer_out)

  const multiplyLayer = tf.layers.multiply();
  const out = multiplyLayer.apply([lstm_out, permuteLayer_out]);


  //skal fjerne en dimension!!
  const lambdaOutput = new lambdaLayer({lambdaFunction: 'result = tf.sum(input[0], 1);', lambdaOutputShape: out.shape}).apply([out])


  const denseLayer1 = tf.layers.dense({units: 6, activation: 'sigmoid'});
  const denseLayer1_out = denseLayer1.apply(lambdaOutput);

  const output = denseLayer1_out;

  const model = tf.model({inputs: input, outputs: output});
  model.summary()
  model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam', metrics: ['accuracy'] });

  //await model.fit(xs, ys, { epochs: epochs });
  //prediction = model.predict(tf.tensor2d([x], [1, 1]));
}
