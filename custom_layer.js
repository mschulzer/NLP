
class lambdaLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
        if (config.name === undefined) {
            config.name = ((+new Date) * Math.random()).toString(36);
        }
        this.name = config.name;
        this.lambdaFunction = config.lambdaFunction;
        this.lambdaOutputShape = config.lambdaOutputShape;
    }

    call(input) {
        return tf.tidy(() => {
            let result = null;
            eval(this.lambdaFunction);
            return result;
        });
    }

    computeOutputShape(inputShape) {
        if (this.lambdaOutputShape === undefined) { //if no outputshape provided, try to set as inputshape
            return inputShape[0];
        } else {
            return this.lambdaOutputShape;
        }
    }

    getConfig() {
        const config = super.getConfig();
        Object.assign(config, {
            lambdaFunction: this.lambdaFunction, lambdaOutputShape: this.lambdaOutputShape
        });
        return config;
    }

    static get className() {
        return 'lambdaLayer';
    }
}
tf.serialization.registerClass(lambdaLayer);
