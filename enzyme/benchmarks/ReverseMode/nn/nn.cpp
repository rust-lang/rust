#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define DADEPT_FLOATING_POINT_TYPE float
#include <adept_source.h>
#include <adept_arrays.h>
using adept::adouble;
using adept::aMatrix;
using adept::aVector;

using adept::Vector;
using adept::adouble;
using adept::aReal;

//extern "C" {
// from https://github.com/AndrewCarterUK/mnist-neural-network-plain-c

#define MNIST_LABEL_MAGIC 0x00000801
#define MNIST_IMAGE_MAGIC 0x00000803
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT
#define MNIST_LABELS 10

typedef struct mnist_label_file_header_t_ {
    uint32_t magic_number;
    uint32_t number_of_labels;
} __attribute__((packed)) mnist_label_file_header_t;

typedef struct mnist_image_file_header_t_ {
    uint32_t magic_number;
    uint32_t number_of_images;
    uint32_t number_of_rows;
    uint32_t number_of_columns;
} __attribute__((packed)) mnist_image_file_header_t;

typedef struct mnist_image_t_ {
    uint8_t pixels[MNIST_IMAGE_SIZE];
} __attribute__((packed)) mnist_image_t;

typedef struct mnist_dataset_t_ {
    mnist_image_t * images;
    uint8_t * labels;
    uint32_t size;
} mnist_dataset_t;

typedef struct neural_network_t_ {
    float b[MNIST_LABELS];
    float W[MNIST_LABELS][MNIST_IMAGE_SIZE];
} neural_network_t;

typedef struct aneural_network_t_ {
    adept::FixedArray<float,true,MNIST_LABELS> b;
    adept::FixedArray<float,true,MNIST_LABELS,MNIST_IMAGE_SIZE> W;
} aneural_network_t;

typedef struct neural_network_gradient_t_ {
    float b_grad[MNIST_LABELS];
    float W_grad[MNIST_LABELS][MNIST_IMAGE_SIZE];
} neural_network_gradient_t;

/**
 * Convert from the big endian format in the dataset if we're on a little endian
 * machine.
 */
uint32_t map_uint32(uint32_t in)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (
        ((in & 0xFF000000) >> 24) |
        ((in & 0x00FF0000) >>  8) |
        ((in & 0x0000FF00) <<  8) |
        ((in & 0x000000FF) << 24)
    );
#else
    return in;
#endif
}

/**
 * Read labels from file.
 * 
 * File format: http://yann.lecun.com/exdb/mnist/
 */
uint8_t * get_labels(const char * path, uint32_t * number_of_labels)
{
    FILE * stream;
    mnist_label_file_header_t header;
    uint8_t * labels;

    stream = fopen(path, "rb");

    if (NULL == stream) {
        fprintf(stderr, "Could not open file: %s\n", path);
        return NULL;
    }

    if (1 != fread(&header, sizeof(mnist_label_file_header_t), 1, stream)) {
        fprintf(stderr, "Could not read label file header from: %s\n", path);
        fclose(stream);
        return NULL;
    }

    header.magic_number = map_uint32(header.magic_number);
    header.number_of_labels = map_uint32(header.number_of_labels);

    if (MNIST_LABEL_MAGIC != header.magic_number) {
        fprintf(stderr, "Invalid header read from label file: %s (%08X not %08X)\n", path, header.magic_number, MNIST_LABEL_MAGIC);
        fclose(stream);
        return NULL;
    }

    *number_of_labels = header.number_of_labels;

    labels = (uint8_t*)malloc(*number_of_labels * sizeof(uint8_t));

    if (labels == NULL) {
        fprintf(stderr, "Could not allocated memory for %d labels\n", *number_of_labels);
        fclose(stream);
        return NULL;
    }

    if (*number_of_labels != fread(labels, 1, *number_of_labels, stream)) {
        fprintf(stderr, "Could not read %d labels from: %s\n", *number_of_labels, path);
        free(labels);
        fclose(stream);
        return NULL;
    }

    fclose(stream);

    return labels;
}

/**
 * Read images from file.
 * 
 * File format: http://yann.lecun.com/exdb/mnist/
 */
mnist_image_t * get_images(const char * path, uint32_t * number_of_images)
{
    FILE * stream;
    mnist_image_file_header_t header;
    mnist_image_t * images;

    stream = fopen(path, "rb");

    if (NULL == stream) {
        fprintf(stderr, "Could not open file: %s\n", path);
        return NULL;
    }

    if (1 != fread(&header, sizeof(mnist_image_file_header_t), 1, stream)) {
        fprintf(stderr, "Could not read image file header from: %s\n", path);
        fclose(stream);
        return NULL;
    }

    header.magic_number = map_uint32(header.magic_number);
    header.number_of_images = map_uint32(header.number_of_images);
    header.number_of_rows = map_uint32(header.number_of_rows);
    header.number_of_columns = map_uint32(header.number_of_columns);

    if (MNIST_IMAGE_MAGIC != header.magic_number) {
        fprintf(stderr, "Invalid header read from image file: %s (%08X not %08X)\n", path, header.magic_number, MNIST_IMAGE_MAGIC);
        fclose(stream);
        return NULL;
    }

    if (MNIST_IMAGE_WIDTH != header.number_of_rows) {
        fprintf(stderr, "Invalid number of image rows in image file %s (%d not %d)\n", path, header.number_of_rows, MNIST_IMAGE_WIDTH);
    }

    if (MNIST_IMAGE_HEIGHT != header.number_of_columns) {
        fprintf(stderr, "Invalid number of image columns in image file %s (%d not %d)\n", path, header.number_of_columns, MNIST_IMAGE_HEIGHT);
    }

    *number_of_images = header.number_of_images;
    images = (mnist_image_t*)malloc(*number_of_images * sizeof(mnist_image_t));

    if (images == NULL) {
        fprintf(stderr, "Could not allocated memory for %d images\n", *number_of_images);
        fclose(stream);
        return NULL;
    }

    if (*number_of_images != fread(images, sizeof(mnist_image_t), *number_of_images, stream)) {
        fprintf(stderr, "Could not read %d images from: %s\n", *number_of_images, path);
        free(images);
        fclose(stream);
        return NULL;
    }

    fclose(stream);

    return images;
}

/**
 * Free all the memory allocated in a dataset. This should not be used on a
 * batched dataset as the memory is allocated to the parent.
 */
void mnist_free_dataset(mnist_dataset_t * dataset)
{
    free(dataset->images);
    free(dataset->labels);
    free(dataset);
}

mnist_dataset_t * mnist_get_dataset(const char * image_path, const char * label_path)
{
    mnist_dataset_t * dataset;
    uint32_t number_of_images, number_of_labels;

    dataset = (mnist_dataset_t*)calloc(1, sizeof(mnist_dataset_t));

    if (NULL == dataset) {
        return NULL;
    }

    dataset->images = get_images(image_path, &number_of_images);

    if (NULL == dataset->images) {
        mnist_free_dataset(dataset);
        return NULL;
    }

    dataset->labels = get_labels(label_path, &number_of_labels);

    if (NULL == dataset->labels) {
        mnist_free_dataset(dataset);
        return NULL;
    }

    if (number_of_images != number_of_labels) {
        fprintf(stderr, "Number of images does not match number of labels (%d != %d)\n", number_of_images, number_of_labels);
        mnist_free_dataset(dataset);
        return NULL;
    }

    dataset->size = number_of_images;

    return dataset;
}

/**
 * Fills the batch dataset with a subset of the parent dataset.
 */
int mnist_batch(mnist_dataset_t * dataset, mnist_dataset_t * batch, int size, int number)
{
    int start_offset;

    start_offset = size * number;

    if (start_offset >= dataset->size) {
        return 0;
    }

    batch->images = &dataset->images[start_offset];
    batch->labels = &dataset->labels[start_offset];
    batch->size = size;

    if (start_offset + batch->size > dataset->size) {
        batch->size = dataset->size - start_offset;
    }

    return 1;
}



#define STEPS 1000
#define BATCH_SIZE 100

#include <stdlib.h>
#include <string.h>
#include <math.h>


// Convert a pixel value from 0-255 to one from 0 to 1
#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)

// Returns a random value between 0 and 1
#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX))

/**
 * Initialise the weights and bias vectors with values between 0 and 1
 */
void neural_network_random_weights(neural_network_t * network)
{
    int i, j;

    for (i = 0; i < MNIST_LABELS; i++) {
        network->b[i] = RAND_FLOAT();

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            network->W[i][j] = RAND_FLOAT();
        }
    }
}

/**
 * Calculate the softmax vector from the activations. This uses a more
 * numerically stable algorithm that normalises the activations to prevent
 * large exponents.
 */
void neural_network_softmax(float * activations, int length)
{
    int i;
    float sum, max;

    for (i = 1, max = activations[0]; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    for (i = 0, sum = 0; i < length; i++) {
        activations[i] = exp(activations[i] - max);
        sum += activations[i];
    }

    for (i = 0; i < length; i++) {
        activations[i] /= sum;
    }
}

/**
 * Calculate the softmax vector from the activations. This uses a more
 * numerically stable algorithm that normalises the activations to prevent
 * large exponents.
 */
void aneural_network_softmax(adept::FixedArray<float,true,MNIST_LABELS> &activations, int length)
{
    int i;
    aReal sum, max;

    for (i = 1, max = activations[0]; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    for (i = 0, sum = 0; i < length; i++) {
        activations[i] = exp(activations[i] - max);
        sum += activations[i];
    }

    activations /= sum;
    for (i = 0; i < length; i++) {
        activations[i] /= sum;
    }
}
static double maxval(const float *activations, int length) {
    float max = activations[0];

    for (int i = 1; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }
    return max;
}

static double sumval(const float *activations, int length) {
    float sum = 0;

    for (int i = 0; i < length; i++) {
        sum += activations[i];
    }
    return sum;
}


static    void makeexps(float* exps, const float* activations, int length, double max) {
    for (int i = 0; i < length; i++) {
        exps[i] = exp(activations[i] - max);
    }

}
/**
 * Calculate the softmax vector from the activations. This uses a more
 * numerically stable algorithm that normalises the activations to prevent
 * large exponents.
 */
static void neural_network_softmax_v2(const float * activations, float* outp, int length)
{
    int i;
    float sum, max;

    for (i = 1, max = activations[0]; i < length; i++) {
        if (activations[i] > max) {
            max = activations[i];
        }
    }

    for (i = 0, sum = 0; i < length; i++) {
        sum += exp(activations[i] - max);
    }

    for (i = 0; i < length; i++) {
        outp[i] = exp(activations[i] - max) / sum;
    }
#if 0
    float max = maxval(activations, length);
    float exps[length];
    makeexps(exps,activations, length, max);
    float sum = sumval(exps,length);
    /*
    for (int i = 0; i < length; i++) {
        double tmp = exps[i];//exp(activations[i] - max);
        sum += tmp;
    }
    */

    for (int i = 0; i < length; i++) {
        outp[i] = exps[i] / sum;
    }
#endif
}
/**
 * Use the weights and bias vector to forward propogate through the neural
 * network and calculate the activations.
 */
void neural_network_hypothesis(const mnist_image_t * image, const neural_network_t * network, float activations[MNIST_LABELS])
{
    int i, j;

    for (i = 0; i < MNIST_LABELS; i++) {
        activations[i] = network->b[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            activations[i] += network->W[i][j] * PIXEL_SCALE(image->pixels[j]);
        }
    }

    neural_network_softmax(activations, MNIST_LABELS);
}

/**
 * Use the weights and bias vector to forward propogate through the neural
 * network and calculate the activations.
 */
static float neural_network_hypothesis_v2(const mnist_image_t * image, const neural_network_t * network, uint8_t label)
{
    float activations[MNIST_LABELS] = {0};
    int i, j;

    for (i = 0; i < MNIST_LABELS; i++) {
        activations[i] = network->b[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            activations[i] += network->W[i][j] * PIXEL_SCALE(image->pixels[j]);
        }
    }

    float activations2[MNIST_LABELS] = { 0 };
    neural_network_softmax_v2(activations, activations2, MNIST_LABELS);
    return -log(activations2[label]);
}


static aReal neural_network_hypothesis_adept(const mnist_image_t * image, const aneural_network_t * network, uint8_t label)
{
    adept::FixedArray<float,true,MNIST_LABELS> activations = network->b;
    int i, j;

    for (i = 0; i < MNIST_LABELS; i++) {
        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            activations(i) += network->W(i,j) * PIXEL_SCALE(image->pixels[j]);
        }
    }

    aneural_network_softmax(activations, MNIST_LABELS);
    return -log(activations[label]);
}


static void calculateDerivatives_adept(mnist_image_t * image, bool run, adept::Stack& stack, const aneural_network_t * anetwork, neural_network_t* gradient, uint8_t label) {

    if (!run) {
        stack.new_recording();
    //    run = true;
    } else
        stack.continue_recording();
    auto resa = neural_network_hypothesis_adept(image, anetwork, label);
    resa.set_gradient(1.0);
    stack.reverse();
    stack.pause_recording();

    for (int i = 0; i < MNIST_LABELS; i++) {
        gradient->b[i] = anetwork->b(i).get_gradient();
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            gradient->W[i][j] = anetwork->W(i,j).get_gradient();
        }
    }
}

extern "C" {
#include <adBuffer.h>
}

void neural_network_softmax_b(float *activations, float *activationsb, int 
        length) {
    float sum, max;
    float sumb, maxb;
    int branch;
    max = activations[0];
    for (int i = 1; i < length; ++i)
        if (activations[i] > max) {
            max = activations[i];
            pushControl1b(1);
        } else
            pushControl1b(0);
    sum = 0;
    for (int i = 0; i < length; ++i) {
        pushReal4(activations[i]);
        activations[i] = (float)exp(activations[i] - max);
        sum = sum + activations[i];
    }
    for (int i = 0; i < length; ++i) {
        pushReal4(activations[i]);
        activations[i] = activations[i]/sum;
    }
    sumb = 0.0;
    for (int i = length-1; i > -1; --i) {
        popReal4(&(activations[i]));
        sumb = sumb - activations[i]*activationsb[i]/(sum*sum);
        activationsb[i] = activationsb[i]/sum;
    }
    {
      float tempb;
      maxb = 0.0;
      for (int i = length-1; i > -1; --i) {
          activationsb[i] = activationsb[i] + sumb;
          popReal4(&(activations[i]));
          tempb = exp(activations[i]-max)*activationsb[i];
          maxb = maxb - tempb;
          activationsb[i] = tempb;
      }
    }
    for (int i = length-1; i > 0; --i) {
        popControl1b(&branch);
        if (branch != 0) {
            activationsb[i] = activationsb[i] + maxb;
            maxb = 0.0;
        }
    }
    activationsb[0] = activationsb[0] + maxb;
}

/**
 * Calculate the softmax vector from the activations. This uses a more
 * numerically stable algorithm that normalises the activations to prevent
 * large exponents.
 */
// Convert a pixel value from 0-255 to one from 0 to 1
// Returns a random value between 0 and 1
void neural_network_softmax_c(float *activations, int length) {
    float sum, max;
    max = activations[0];
	int i;
    for (i = 1; i < length; ++i)
        if (activations[i] > max)
            max = activations[i];
    sum = 0;
    for (i = 0; i < length; ++i) {
        activations[i] = (float)exp(activations[i] - max);
        sum += activations[i];
    }
    for (i = 0; i < length; ++i)
        activations[i] /= sum;
}

/*
  Differentiation of neural_network_hypothesis_tapenadesource in reverse (adjoint) mode:
   gradient     of useful results: neural_network_hypothesis_tapenadesource
                *network.b[0:10-1] *network.W[0:10-1][0:28*28-1]
   with respect to varying inputs: *network.b[0:10-1] *network.W[0:10-1][0:28*28-1]
   RW status of diff variables: neural_network_hypothesis_tapenadesource:in-killed
                *network.b[0:10-1]:incr *network.W[0:10-1][0:28*28-1]:incr
   Plus diff mem management of: network:in *network.b:in *network.W:in
                *network.W[0:10-1]:in
*/
static void neural_network_hypothesis_tapenadesource_b(const mnist_image_t *
        image, const neural_network_t *network, neural_network_t *networkb, 
        uint8_t label, float neural_network_hypothesis_tapenadesourceb) {
    float activations[10];
    float activationsb[10];
    int ii1;
    float neural_network_hypothesis_tapenadesource;
    for (int i = 0; i < 10; ++i) {
        activations[i] = network->b[i];
        for (int j = 0; j < 784; ++j)
            activations[i] = activations[i] + network->W[i][j]*((float)image->
                pixels[j]/255.0f);
    }
    pushReal4Array(activations, 10);
    neural_network_softmax_c(activations, 10);
    for (ii1 = 0; ii1 < 10; ++ii1)
        activationsb[ii1] = 0.0;
    activationsb[(int)label] = activationsb[(int)label] - 
        neural_network_hypothesis_tapenadesourceb/activations[(int)label];
    popReal4Array(activations, 10);
    neural_network_softmax_b(activations, activationsb, 10);
    for (int i = 9; i > -1; --i) {
        for (int j = 783; j > -1; --j)
            networkb->W[i][j] = networkb->W[i][j] + (float)image->pixels[j]*
                activationsb[i]/255.0f;
        networkb->b[i] = networkb->b[i] + activationsb[i];
        activationsb[i] = 0.0;
    }
}

/**
 * Update the gradients for this step of gradient descent using the gradient
 * contributions from a single training example (image).
 * 
 * This function returns the loss ontribution from this training example.
 */
float neural_network_gradient_update(mnist_image_t * image, const neural_network_t * network, neural_network_gradient_t * gradient, uint8_t label)
{
    float activations[MNIST_LABELS];
    float b_grad, W_grad;
    int i, j;

    // First forward propagate through the network to calculate activations
    neural_network_hypothesis(image, network, activations);

    for (i = 0; i < MNIST_LABELS; i++) {
        // This is the gradient for a softmax bias input
        b_grad = (i == label) ? activations[i] - 1 : activations[i];

        for (j = 0; j < MNIST_IMAGE_SIZE; j++) {
            // The gradient for the neuron weight is the bias multiplied by the input weight
            W_grad = b_grad * PIXEL_SCALE(image->pixels[j]);

            // Update the weight gradient
            gradient->W_grad[i][j] += W_grad;
        }

        // Update the bias gradient
        gradient->b_grad[i] += b_grad;
    }

    // Cross entropy loss
    return 0.0f - log(activations[label]);
}


extern int enzyme_const;
template<typename Return, typename... T>
Return __enzyme_autodiff(T...);

static void calculateDerivatives(mnist_image_t * image, const neural_network_t * network, neural_network_t* gradient, uint8_t label) {
    __enzyme_autodiff<void>(neural_network_hypothesis_v2, enzyme_const, image, network, gradient, enzyme_const, label);
}

/**
 * Run one step of gradient descent and update the neural network.
 */
float neural_network_training_step(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate)
{
    neural_network_t gradient = {0};
    neural_network_t gradient2 = {0};

    /*
    adept::Stack stack;
    aneural_network_t anetwork;
  
    for (int i = 0; i < MNIST_LABELS; i++) {
        anetwork.b[i] = network->b[i];
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            anetwork.W[i][j] = network->W[i][j];
        }
    }*/

    float total_loss;
    int i, j;

    // Calculate the gradient and the loss by looping through the training set
    for (i = 0, total_loss = 0; i < dataset->size; i++) {
		mnist_image_t* image = &dataset->images[i];
		uint8_t label = dataset->labels[i];

    	// First forward propagate through the network to calculate activations

        //calculateDerivatives(image, network, &gradient, label);
        //calculateDerivatives_adept(image, i != 0, stack, &anetwork, &gradient, label);
		//neural_network_hypothesis_tapenadesource_b(image, network, &gradient, label, 1.0);

        total_loss +=neural_network_gradient_update(image, network, (neural_network_gradient_t*)&gradient, label);
        
        //total_loss +=neural_network_gradient_update(image, network, (neural_network_gradient_t*)&gradient2, label);

	    //float activations[MNIST_LABELS];
        //neural_network_hypothesis(image, network, activations);
    	//total_loss -= log(activations[label]);

    }

    // Apply gradient descent to the network
    for (i = 0; i < MNIST_LABELS; i++) {
        //printf("b'[i] %f %f\n", gradient.b[i], gradient2.b[i]);
        network->b[i] -= learning_rate * gradient.b[i] / ((float) dataset->size);

        for (j = 0; j < MNIST_IMAGE_SIZE + 1; j++) {
            network->W[i][j] -= learning_rate * gradient.W[i][j] / ((float) dataset->size);
        }
    }

    return total_loss;
}

/**
 * Run one step of gradient descent and update the neural network.
 */
float neural_network_training_step_enzyme(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate)
{
    neural_network_t gradient = {0};
    neural_network_t gradient2 = {0};

    float total_loss;
    int i, j;

    // Calculate the gradient and the loss by looping through the training set
    for (i = 0, total_loss = 0; i < dataset->size; i++) {
		mnist_image_t* image = &dataset->images[i];
		uint8_t label = dataset->labels[i];

    	// First forward propagate through the network to calculate activations

        calculateDerivatives(image, network, &gradient, label);
        //calculateDerivatives_adept(image, i != 0, stack, &anetwork, &gradient, label);
		//neural_network_hypothesis_tapenadesource_b(image, network, &gradient, label, 1.0);
 
        //total_loss +=neural_network_gradient_update(image, network, (neural_network_gradient_t*)&gradient2, label);

	    float activations[MNIST_LABELS];
        neural_network_hypothesis(image, network, activations);
    	total_loss -= log(activations[label]);

    }

    // Apply gradient descent to the network
    for (i = 0; i < MNIST_LABELS; i++) {
        //printf("b'[i] %f %f\n", gradient.b[i], gradient2.b[i]);
        network->b[i] -= learning_rate * gradient.b[i] / ((float) dataset->size);

        for (j = 0; j < MNIST_IMAGE_SIZE + 1; j++) {
            network->W[i][j] -= learning_rate * gradient.W[i][j] / ((float) dataset->size);
        }
    }

    return total_loss;
}

/**
 * Run one step of gradient descent and update the neural network.
 */
float neural_network_training_step_adept(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate)
{
    neural_network_t gradient = {0};
    neural_network_t gradient2 = {0};

    adept::Stack stack;
    aneural_network_t anetwork;
  
    for (int i = 0; i < MNIST_LABELS; i++) {
        anetwork.b[i] = network->b[i];
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            anetwork.W[i][j] = network->W[i][j];
        }
    }

    float total_loss;
    int i, j;

    // Calculate the gradient and the loss by looping through the training set
    for (i = 0, total_loss = 0; i < dataset->size; i++) {
		mnist_image_t* image = &dataset->images[i];
		uint8_t label = dataset->labels[i];

    	// First forward propagate through the network to calculate activations

        calculateDerivatives_adept(image, i != 0, stack, &anetwork, &gradient, label);
        
        //total_loss +=neural_network_gradient_update(image, network, (neural_network_gradient_t*)&gradient2, label);

	    float activations[MNIST_LABELS];
        neural_network_hypothesis(image, network, activations);
    	total_loss -= log(activations[label]);

    }

    // Apply gradient descent to the network
    for (i = 0; i < MNIST_LABELS; i++) {
        //printf("b'[i] %f %f\n", gradient.b[i], gradient2.b[i]);
        network->b[i] -= learning_rate * gradient.b[i] / ((float) dataset->size);

        for (j = 0; j < MNIST_IMAGE_SIZE + 1; j++) {
            network->W[i][j] -= learning_rate * gradient.W[i][j] / ((float) dataset->size);
        }
    }

    return total_loss;
}

/**
 * Run one step of gradient descent and update the neural network.
 */
float neural_network_training_step_tapenade(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate)
{
    neural_network_t gradient = {0};
    neural_network_t gradient2 = {0};

    float total_loss;
    int i, j;

    // Calculate the gradient and the loss by looping through the training set
    for (i = 0, total_loss = 0; i < dataset->size; i++) {
		mnist_image_t* image = &dataset->images[i];
		uint8_t label = dataset->labels[i];

    	// First forward propagate through the network to calculate activations

		neural_network_hypothesis_tapenadesource_b(image, network, &gradient, label, 1.0); 
        //total_loss +=neural_network_gradient_update(image, network, (neural_network_gradient_t*)&gradient2, label);

	    float activations[MNIST_LABELS];
        neural_network_hypothesis(image, network, activations);
    	total_loss -= log(activations[label]);
    }

    // Apply gradient descent to the network
    for (i = 0; i < MNIST_LABELS; i++) {
        //printf("b'[i] %f %f\n", gradient.b[i], gradient2.b[i]);
        network->b[i] -= learning_rate * gradient.b[i] / ((float) dataset->size);

        for (j = 0; j < MNIST_IMAGE_SIZE + 1; j++) {
            network->W[i][j] -= learning_rate * gradient.W[i][j] / ((float) dataset->size);
        }
    }

    return total_loss;
}

/**
 * Downloaded from: http://yann.lecun.com/exdb/mnist/
 */
const char * train_images_file = "data/train-images-idx3-ubyte";
const char * train_labels_file = "data/train-labels-idx1-ubyte";
const char * test_images_file = "data/t10k-images-idx3-ubyte";
const char * test_labels_file = "data/t10k-labels-idx1-ubyte";

/**
 * Calculate the accuracy of the predictions of a neural network on a dataset.
 */
float calculate_accuracy(mnist_dataset_t * dataset, neural_network_t * network) {
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++) {
        // Calculate the activations for each image using the neural network
        neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float) correct) / ((float) dataset->size);
}

#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

void run(float (*fn)(mnist_dataset_t*, neural_network_t*, float)) {
    mnist_dataset_t * train_dataset, * test_dataset;
    mnist_dataset_t batch;
    neural_network_t network;
    float loss, accuracy;
    int i, batches;

    // Read the datasets from the files
    train_dataset = mnist_get_dataset(train_images_file, train_labels_file);
    test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

    // Initialise weights and biases with random values
    neural_network_random_weights(&network);

    // Calculate how many batches (so we know when to wrap around)
    batches = train_dataset->size / BATCH_SIZE;

    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    for (i = 0; i < STEPS; i++) {
        // Initialise a new batch
        mnist_batch(train_dataset, &batch, 100, i % batches);

        // Run one step of gradient descent and calculate the loss
        loss = fn(&batch, &network, 0.5);

        // Calculate the accuracy using the whole test dataset
        accuracy = calculate_accuracy(test_dataset, &network);

        printf("Step %04d\tAverage Loss: %.2f\tAccuracy: %.3f\n", i, loss / batch.size, accuracy);
    }
  
    gettimeofday(&end, NULL);
    printf("%0.6f\n", tdiff(&start, &end));

    // Cleanup
    mnist_free_dataset(train_dataset);
    mnist_free_dataset(test_dataset);

}

int main(int argc, char *argv[])
{
    printf("Regular\n");
    run(neural_network_training_step);
    printf("Enzyme\n");
    run(neural_network_training_step_enzyme);
    printf("Adept\n");
    run(neural_network_training_step_adept);
    printf("Tapenade\n");
    run(neural_network_training_step_tapenade);
    return 0;
}
