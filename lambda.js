class MyLambda extends tf.layers.Layer {
  constructor() {
    super({});

  }

  computeOutputShape(inputShape) {
    return inputShape
  }

  /*
   * @param inputs Tensor to be treated.
   * @param kwargs Only used as a pass through to call hooks.
   */
  call(inputs, kwargs) {
    let input = inputs;
    if (Array.isArray(input)) {
      input = input[0];
    }

    return tf.sum(inputs[0], 1)
  }

  /**
   * If a custom layer class is to support serialization, it must implement
   * the `className` static getter.
   */
  static get className() {
    return 'MyLambda';
  }
}
tf.serialization.registerClass(MyLambda);  // Needed for serialization.

export function mylambda() {
  return new MyLambda();
}
