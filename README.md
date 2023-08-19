# Multi-Language_LSTM_Cell_Implementations
Simple LSTM cell structures implemented in multiple langauges. 

## Contents 

# C++ LSTM (Experiment)
The following repository is a simple LSTM network implemented in C++. 
This is an ongoing experiment, and not intended as a reliable or complete package. 
The LSTM is constructed as follows. 

## Questions I havent answered:
* how do I use a LSTM to equalise a Rayleigh channel, how is the LSTM trained and updated?
* Why is this method superior to LS and MMSE estimation? How can it be superior if it is trained my LS and MMSE?

## LSTM Cell Architecture

![LSTM Architecture](https://github.com/Meandi-n/Cpp-LSTM/blob/main/LSTM_draw.drawio.png)

## LSTM Network Architecture

![LSTM Network](https://github.com/ryan-n-may/Cpp-LSTM/blob/main/LSTM_network.png)

## Forward Propagation

Forward propagation is implemented by Cell.h in void forwardPropagation(void). 
```c++
void Cell::forwardPropagation(void)
{
    Operations::dimension wd;
    wd.d1 = Cell::input_length;
    wd.d2 = Cell::input_length;
    // input gate
    Cell::Zg = Operations::multiplyWithWeights(Cell::Wxg, Cell::xt, wd);
    Cell::Zg = Operations::sumVectors(Cell::Zg, Operations::multiplyWithWeights(Cell::Whg, Cell::htm1, wd), Cell::input_length);
    Cell::Zg = Operations::sumVectors(Cell::Zg, Cell::bg, Cell::input_length);
    Cell::gt = Operations::tanh_vector(Cell::Zg, Cell::input_length);
    Cell::Zi = Operations::multiplyWithWeights(Cell::Wxi, Cell::xt, wd);
    Cell::Zi = Operations::sumVectors(Cell::Zi, Operations::multiplyWithWeights(Cell::Whi, Cell::htm1, wd), Cell::input_length);
    Cell::Zi = Operations::sumVectors(Cell::Zi, Cell::bi, Cell::input_length);
    Cell::it = Operations::sigmoid(Cell::Zi, Cell::input_length);
    Cell::i_gate = Operations::multiplyVectors(Cell::gt, Cell::input_length, Cell::it, Cell::input_length);
    // forget fate
    Cell::Zf = Operations::multiplyWithWeights(Cell::Wxf, Cell::xt, wd);
    Cell::Zf = Operations::sumVectors(Cell::Zf, Operations::multiplyWithWeights(Cell::Whf, Cell::htm1, wd), Cell::input_length);
    Cell::Zf = Operations::sumVectors(Cell::Zf, Cell::bf, Cell::input_length);
    Cell::ft = Operations::sigmoid(Cell::Zf, Cell::input_length);
    Cell::f_gate = Cell::ft;
    // output gate
    Cell::Zo = Operations::multiplyWithWeights(Cell::Wxo, Cell::xt, wd);
    Cell::Zo = Operations::sumVectors(Cell::Zo, Operations::multiplyWithWeights(Cell::Who, Cell::htm1, wd), Cell::input_length);
    Cell::Zo = Operations::sumVectors(Cell::Zo, Cell::bg, Cell::input_length);
    Cell::ot = Operations::sigmoid(Cell::Zo, Cell::input_length);
    Cell::o_gate = Cell::ot;
    // current cell state
    Cell::ct = Operations::multiplyVectors(Cell::ctm1, Cell::input_length, Cell::f_gate, Cell::input_length);
    Cell::ct = Operations::sumVectors(Cell::ct, Cell::i_gate, Cell::input_length);
    // prdiction state
    Cell::ht = Operations::tanh_vector(Cell::ct, Cell::input_length);
    Cell::ht = Operations::multiplyVectors(Cell::ht, Cell::input_length, Cell::o_gate, Cell::input_length);
}
```
$$Z_g = W_{xg} x_t + W_{hg} h_{t-1} + b_g $$

$$g_t = tanh(Z_g) $$

$$Z_i = W_{xi} x_t + W_{hi} h_{t-1} + b_i $$

$$i_t = sigmoid(Z_i) $$

$$i_{gate} = g_t i_t $$

$$Z_f = W_{xf} x_t + W_{hf} h_{t-1} + b_f $$ 

$$f_t = sigmoid(Z_f) $$

$$f_{gate} = f_t $$

$$Z_o = W_{xg} x_t + W_{hg} h_{t-1} + b_g $$

$$o_t = sigmoid(Z_o) $$

$$o_{gate} = o_t $$

$$c_t = (c_{t-1}f_{gate}) + i_{gate} $$

$$h_t = o_{gate} tanh(c_t) $$

## BPTT calculate gradients
```c++
void Cell::gradientCalculation(void)
{
    // Error of cell
    Cell::dE = Operations::subVectors(Cell::yt, Cell::ct, Cell::input_length);
    /*Gradient with respect to gates*/
    Cell::dEdot = Operations::tanh_vector(Cell::ct, Cell::input_length);
    Cell::dEdot = Operations::multiplyVectors(Cell::dEdot, Cell::input_length, Cell::dE, Cell::input_length);

    Cell::dEdct = Operations::invtanh_vector(Cell::ct, Cell::input_length);
    Cell::dEdct = Operations::multiplyVectors(Cell::dEdct, Cell::input_length, Cell::ot, Cell::input_length);
    Cell::dEdct = Operations::multiplyVectors(Cell::dEdct, Cell::input_length, Cell::dE, Cell::input_length);

    Cell::dEdit = Operations::multiplyVectors(Cell::dEdct, Cell::input_length, Cell::it, Cell::input_length);

    Cell::dEdft = Operations::multiplyVectors(Cell::dEdct, Cell::input_length, Cell::ctm1, Cell::input_length);

    Cell::dEdctm1 = Operations::multiplyVectors(Cell::dEdct, Cell::input_length, Cell::ft, Cell::input_length);

    /*Gradient with respect to output weights*/
    Cell::dEdbo = Operations::multiplyVectors(Cell::dE, Cell::input_length, Operations::tanh_vector(Cell::ct, Cell::input_length), Cell::input_length);
    Cell::dEdbo = Operations::multiplyVectors(Cell::dEdbo, Cell::input_length, Operations::sigmoid(Cell::Zo, Cell::input_length), Cell::input_length);
    Cell::dEdbo = Operations::multiplyVectors(Cell::dEdbo, Cell::input_length, Operations::invsigmoid(Cell::Zo, Cell::input_length), Cell::input_length);

    Cell::dEdWxo = Operations::multiplyVectors(Cell::dEdbo, Cell::input_length, Cell::xt, Cell::input_length);
    Cell::dEdWho = Operations::multiplyVectors(Cell::dEdbo, Cell::input_length, Cell::htm1, Cell::input_length);

    /*Gradient with respect to forget weights*/
    Cell::dEdbf = Operations::multiplyVectors(Cell::dE, Cell::input_length, Operations::invtanh_vector(Cell::ct, Cell::input_length), Cell::input_length);
    Cell::dEdbf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::ot, Cell::input_length);
    Cell::dEdbf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::ctm1, Cell::input_length);
    Cell::dEdbf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Operations::sigmoid(Cell::Zf, Cell::input_length), Cell::input_length);
    Cell::dEdbf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Operations::invsigmoid(Cell::Zf, Cell::input_length), Cell::input_length);

    Cell::dEdWxf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::xt, Cell::input_length);
    Cell::dEdWhf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::htm1, Cell::input_length);

    /*Gradient with respect to input weights*/
    Cell::dEdbi = Operations::multiplyVectors(Cell::dE, Cell::input_length, Operations::invtanh_vector(Cell::ct, Cell::input_length), Cell::input_length);
    Cell::dEdbi = Operations::multiplyVectors(Cell::dEdbi, Cell::input_length, Cell::ot, Cell::input_length);
    Cell::dEdbi = Operations::multiplyVectors(Cell::dEdbi, Cell::input_length, Cell::gt, Cell::input_length);
    Cell::dEdbi = Operations::multiplyVectors(Cell::dEdbi, Cell::input_length, Operations::sigmoid(Cell::Zi, Cell::input_length), Cell::input_length);
    Cell::dEdbi = Operations::multiplyVectors(Cell::dEdbi, Cell::input_length, Operations::invsigmoid(Cell::Zi, Cell::input_length), Cell::input_length);

    Cell::dEdWxi = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::xt, Cell::input_length);
    Cell::dEdWhi = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::htm1, Cell::input_length);

    /*Gradient with respect to c and g states*/
    Cell::dEdbg = Operations::multiplyVectors(Cell::dE, Cell::input_length, Cell::ot, Cell::input_length);
    Cell::dEdbg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Operations::invtanh_vector(Cell::ct, Cell::input_length), Cell::input_length);
    Cell::dEdbg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Cell::ot, Cell::input_length);
    Cell::dEdbg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Cell::it, Cell::input_length);
    Cell::dEdbg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Operations::invtanh_vector(Cell::Zg, Cell::input_length), Cell::input_length);

    Cell::dEdWhg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Cell::htm1, Cell::input_length);
    Cell::dEdWxg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Cell::xt, Cell::input_length);
}
```
#### Gradient with respect to gates
$$ dE = y_t - h_t $$

$$ dE/do_t = dE * tanh(c_t) $$

$$ dE/dc_t = dE * o_t * (1-tanh(c_t)^2) $$

$$ dE/di_t = dE * o_t * (1-tanh(c_t)^2) * i_t $$

$$ dE/df_t = dE * o_t * (1-tanh(c_t)^2) * c_{t-1} $$

$$ dE/dc_{t-1} = dE * o_t * (1-tanh(c_t)^2) * f_t $$

#### Gradeint with respect to output weights 
$$ dE/dW_{xo} = dE * tanh(c_t) * sigmoid(Z_o) * (1-sigmoid(Z_o)) * x_t $$

$$ dE/dW_{ho} = dE * tanh(c_t) * sigmoid(Z_o) * (1-sigmoid(Z_o)) * h_{t-1} $$

$$ dE/db_{o} = dE * tanh(c_t) * sigmoid(Z_o) * (1-sigmoid(Z_o)) $$

#### Gradient with respect to forget weights
$$ dE/dW_{xf} = dE * o_t * (1-tanh(c_t)^2) * c_{t-1} * sigmoid(Z_f) * (1-sigmoid(Z_f)) * x_t $$

$$ dE/dW_{hf} = dE * o_t * (1-tanh(c_t)^2) * c_{t-1} * sigmoid(Z_f) * (1-sigmoid(Z_f)) * h_{t-1}$$

$$ dE/db_{f} = dE * o_t * (1-tanh(c_t)^2) * c_{t-1} * sigmoid(Z_f) * (1-sigmoid(Z_f)) $$

#### Gradient with respect to input weights
$$ dE/dW_{xi} = dE * o_t * (1-tanh(c_t)^2) * g_{t} * sigmoid(Z_i) * (1-sigmoid(Z_i)) * x_t $$

$$ dE/dW_{hi} = dE * o_t * (1-tanh(c_t)^2) * g_{t} * sigmoid(Z_i) * (1-sigmoid(Z_i)) * h_{t-1}$$

$$ dE/db_{i} = dE * o_t * (1-tanh(c_t)^2) * g_{t} * sigmoid(Z_i) * (1-sigmoid(Z_i)) $$

#### Gradient with respect to c and g gates
$$ dE/dW_{xg} = dE * o_t * (1-tanh(c_t)^2) * i_{t} * (1-tanh(Z_g)^2) * x_t $$

$$ dE/dW_{hg} = dE * o_t * (1-tanh(c_t)^2) * i_{t} * (1-tanh(Z_g)^2) * h_{t-1} $$

$$ dE/db_{g} = dE * o_t * (1-tanh(c_t)^2) * i_{t} * (1-tanh(Z_g)^2) $$

## Updating weights  

Alpha is the learning rate of the LSTM, configured when creating the LSTM 
network. 

$$ W_{xo} <= W_{xo} + dE/dW_{xo} * α $$

$$ W_{ho} <= W_{ho} + dE/dW_{ho} * α $$

$$ W_{xf} <= W_{xf} + dE/dW_{xf} * α $$

$$ W_{hf} <= W_{hf} + dE/dW_{hf} * α $$

$$ W_{xi} <= W_{xi} + dE/dW_{xi} * α $$

$$ W_{hi} <= W_{hi} + dE/dW_{hi} * α $$

$$ W_{xg} <= W_{xg} + dE/dW_{xg} * α $$

$$ W_{hg} <= W_{hg} + dE/dW_{hg} * α $$


## Updating bias vectors

$$ b_{o} <= b_{o} + dE/db_{o} $$

$$ b_{f} <= b_{f} + dE/db_{f} $$

$$ b_{i} <= b_{i} + dE/db_{i} $$

$$ b_{g} <= b_{g} + dE/db_{g} $$

# Single cell training test
Single LSTM cells can be trained through BPTT to output a target (y) by adjusting weights and biases for a given input (x).  This means that the cell is specialised to produce y for given input x. This provides a good testing/ proof of concept for the LSTM cell. 
## Test 1: input length = 4
```
Input x: {1,1,1,1} 
Output y: {1,2,3,4} (target) 
initial cell states are set to vectors of 0. 
```
The following plot shows how each round brings input elements (1 to 4) closer to the target elements.  Each progression along the x axis is another round of LSTM training.  
![Single cell test length 4](https://github.com/Meandi-n/Cpp-LSTM/blob/main/single_cell_test_length4.png)

## Test 2: input length = 20
```
Input x: {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}
Output y: {1,2,3,4,5,6,9,3,4,5,4,2,3,1,4,4,4,1,1,1} (target) 
initial cell states are set to vectors of 0. 
```
The following plot shows how each round brings input elements (1 to 20) closer to the target elements.  Each progression along the x axis is another round of LSTM training.  
![Single cell test length 20](https://github.com/Meandi-n/Cpp-LSTM/blob/main/single_cell_test_length_20.png)

# LSTM Network Testing

an LSTM network can be trained to a Rayleigh channel of length 64, using 32 LSTM cells of input dimension 2. 

```c++
int _input_length = 2;
float _weight_range = 1;
float _learning_rate = 1;
int _network_size = 32;
Network* net;
```
The LS channel is given below.  The LSTM is trained by setting initial cell state to {0,0}, and expecting a row of LS_EST as the output of each LSTM cell.  Weights and biases are configured such that LSTM output matches LS_EST (LS channel estimation). 
```c++
 vector<float> LS_EST = {-1.51120275010916000,	-2.04978774403636000,
                         -1.86959755782467000,  -1.28733708601674000,
                         -0.61971122315524300,	-0.14377093011389100,
                          0.02204856690934480,	-0.13848543119575400,
                         -0.39141124986415700,  -0.49386791995177700,
                         -0.41759166767354000,	-0.21386033162764900,
                         -0.03752100458637800,	-0.02388244855140030,
                         -0.12653084826790100,  -0.25309853150552400,
                         -0.30912497538824300,	-0.23405155789833900,
                         -0.10502011109699700,	-0.01996145279267770,
                         -0.02226188970404040,  -0.11725570681897900,
                         -0.21261361329212200,	-0.21794283471779400,
                         -0.14826023828236900,	-0.05017635940264710,
                         0.008733235174919100,  -0.02211203113601590,
                         -0.10434774295170800,	-0.17853723838995900,
                         -0.18989986154485700,	-0.10970779864006500,
                         -0.00878859844129037,  +0.03270392235059590,
                         -0.00313376216390587,	-0.10040833092472600,
                         -0.17028936583489300,	-0.13543326525558300,
                         -0.03738079041746790,  +0.05377053278093730,
                         +0.07258048263924660,	-0.01476863074222770,
                         -0.12026167425850800,	-0.14697029531602000,
                         -0.08414608637146880,  +0.04104308287672750,
                         +0.13594417881357900,	+0.11900784845341700,
                         +0.01858462524442080,	-0.09979946704941600,
                         -0.13181632983469800,  -0.00657875501899657,
                         +0.17423482865685000,	+0.27988981506781900,
                         +0.23599252538542600,	+0.03213783127723470,
                         -0.14246612090153200,  -0.09543037611829610,
                         +0.17874398068126800,	+0.60059504067689400,
                         +0.93770763952819200,	+0.91942829901473900,
                         +0.27510354091611400,  -1.26592011298811000 };
```
```c++
int ROUNDS = 2000;
network_test::net->trainNetwork(network_test::ROUNDS);
```
The Network is then tested
```c++
/** Test network **/
network_test::net->testNetwork({0,0});
Operations::save2DVector(network_test::net->predictions, 32, 2, "testing_VECTOR.csv");
````
The LS channel then changes slightly, in order for the LSTM to model the network, it must be retrained.  However, due to the changes of the LS network being small, only 50 training rounds are required, rather than 2000. 
```c++
network_test::net->trainNetwork(50);
Operations::save2DVector(network_test::net->predictions, 32, 2, "predictions2_VECTOR.csv");
```
## MatLab plot showing training of original, and adjusted LS estimation

![LSTM plot](https://github.com/ryan-n-may/Cpp-LSTM/blob/main/LSTM_network_plot.png)

# LSTM-Cell-Ruby
This repository is an LSTM cell (single cell) simulated in RUBY.
It is a building block of a complete LSTM to be developed further.

## LSTM Architecture
The following architecure is implemented in Ruby.
![LSTM diagram](https://github.com/ryan-n-may/LSTM-Cpp/blob/main/LSTM_draw.drawio.png)

## Forward Propagation
$$Z_g = W_{xg} x_t + W_{hg} h_{t-1} + b_g $$

$$g_t = tanh(Z_g) $$

$$Z_i = W_{xi} x_t + W_{hi} h_{t-1} + b_i $$

$$i_t = sigmoid(Z_i) $$

$$i_{gate} = g_t i_t $$

$$Z_f = W_{xf} x_t + W_{hf} h_{t-1} + b_g $$ 

$$f_t = sigmoid(Z_f) $$

$$f_{gate} = f_t $$

$$Z_o = W_{xo} x_t + W_{ho} h_{t-1} + b_g $$

$$o_t = sigmoid(Z_o) $$

$$o_{gate} = o_t $$

$$c_t = (c_{t-1}f_{gate}) + i_{gate} $$

$$h_t = o_{gate} tanh(c_t) $$

```ruby
def forwardPropagation()
  # 'g' operations
	@Zg = multiplyWithWeights(@Xtm1, @Wxg) + multiplyWithWeights(@Htm1, @Whg) + @Bg.transpose()
	@Gt = tanhVector(@Zg)
	# 'i' operations
	@Zi = multiplyWithWeights(@Xtm1, @Wxi) + multiplyWithWeights(@Htm1, @Whi) + @Bi.transpose()
	@It = sigmoidVector(@Zi)
	@Igate = @Gt *~ @It
	# 'f' operations
	@Zf = multiplyWithWeights(@Xtm1, @Wxf) + multiplyWithWeights(@Htm1, @Whf) + @Bf.transpose()
	@Ft = sigmoidVector(@Zf)
	@Fgate = @Ft
	# 'o' operations
	@Zo = multiplyWithWeights(@Xtm1, @Wxo) + multiplyWithWeights(@Htm1, @Who) + @Bg.transpose()
	@Ot = sigmoidVector(@Zo)
	@Ogate = @Ot
	# cell state operations
	@Ct = (@Ctm1 *~ @Fgate) + @Igate
	@Ht = @Ogate *~ tanhVector(@Ct)
end
```

## Backward propagation
```ruby
def backwardPropagation()
  # Calculating the gradients with respect to weights and output error
	# (see github for explanation in C++)
	# https://github.com/ryan-n-may/LSTM-Cpp/tree/main
	dE = @Yt - @Ht
	# Gradient with respect to gates and states
	dE_dot   = dE *~ tanhVector(@Ct)
	dE_dct   = dE *~ @Ot *~ invTanhVector(@Ct)
	dE_dit   = dE_dct *~ @It
	dE_dft   = dE_dct *~ @Ctm1
	dE_dctm1 = dE_dct *~ @Ft
	# Gradient with respect to output weights
	dE_dbo = dE *~ tanhVector(@Ct) *~ sigmoidVector(@Zo) *~ invSigmoidVector(@Zo)
	dE_dWxo = dE_dbo *~ @Xtm1 
	dE_dWho = dE_dbo *~ @Htm1
	# Gradient with respect to forget weights
	dE_dbf  = dE *~ @Ot *~ invTanhVector(@Ct) *~ @Ctm1 *~ sigmoidVector(@Zf) *~ invSigmoidVector(@Zf)
	dE_dWxf = dE_dbf *~ @Xtm1
	dE_dWhf = dE_dbf *~ @Htm1
	# Gradient with respect to input weights
	dE_dbi  = dE *~ @Ot *~ invTanhVector(@Ct) *~ @Gt *~ sigmoidVector(@Zi) *~ invSigmoidVector(@Zi)
	dE_dWxi = dE_dbi *~ @Xtm1
	dE_dWhi = dE_dbi *~ @Htm1
	# Gradient with respect to cell states @Gt
	dE_dbg  = dE *~ @Ot *~ @Ot *~ invTanhVector(@Ct) *~ @It *~ invTanhVector(@Zg) 
	dE_dWxg = dE_dbg *~ @Xtm1
	dE_dWhg = dE_dbg *~ @Htm1
	# Now we update the weights using the calculated gradients.
	# Modifying output weights
	@Wxo = updateWeights(@Wxo, dE_dWxo, @@Alpha)
	@Who = updateWeights(@Who, dE_dWho, @@Alpha)
	# Modifying forget weights	
	@Wxf = updateWeights(@Wxf, dE_dWxf, @@Alpha)
	@Whf = updateWeights(@Whf, dE_dWhf, @@Alpha)
	# Modifying input weights
	@Wxi = updateWeights(@Wxi, dE_dWxi, @@Alpha)
	@Whi = updateWeights(@Whi, dE_dWhi, @@Alpha)
	# Modifying g weights
	@Wxg = updateWeights(@Wxg, dE_dWxg, @@Alpha)
	@Whg = updateWeights(@Whg, dE_dWhg, @@Alpha)
	# Modifying bias vectors
	@Bf = @Bf + dE_dbf.transpose() # UNSURE IF THESE SHOULD ADD OR MINUS
	@Bi = @Bi + dE_dbi.transpose()
	@Bg = @Bg + dE_dbg.transpose()
end
```
## Weight modification
The figure here shows the weight matrices before traning (left) and after training (right)

![output](https://github.com/ryan-n-may/LSTM-Cell-Ruby/blob/main/output.png)

# One-hot-dictionary in Ruby
This Ruby gem creates an encoder and dictionary class that are capable of one-hot encoding characters and strings from input sentances. 

![ruby](https://img.shields.io/badge/Ruby-CC342D?style=for-the-badge&logo=ruby&logoColor=white) 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Usage
### Installation 
```ruby
gem install one-hot-dictionary
require 'one-hot-dictionary'
```
* requires Numo::NArrray https://github.com/ruby-numo/numo-narray
### Documentation
```ruby
@dict = DICTIONARY.new
```
#### initialisation
```ruby
@dict.init(input_message [string], x_dim [integer])
```
* This method generates a dictionary encoding of size x_dim.
* If words in the input_message outnumber x_dim, words are prioritised by the highest frequency.
#### Get encoding and decoding hash 
```ruby
@dict.getEncodingHashs() -> @stringToEncoding, @encodingToString [hash]
```
* `@stringToEncoding` maps dictionary words [string] to binary [Numo::NArray].
* `@encodingToString` maps binary [Numo::NArray] to words [string]
#### Get frequency hash
```ruby
@dict.getFrequencyHash() -> @frequencyHash [hash]
```
* `@frequencyHash` maps all words of the `input_message` to their occurances. 
#### Word frequency
```ruby
@dict.wordFrequency(input_message [string or []string]) -> frequency [hash]
```
* Converts input string or array of strings into hash of occurance counts, the counts are sorted in descending order.
#### Hot encode vocabulary
```ruby
@dict.hotEncodeVocabulary(frequency [hash], maxEntries [integer]) -> stringToEncoding, encodingToString [hash]
```
* Encodes vocabulary words from frequency hash. Limits vocabulary by `maxEntries`.
* ie: `dog` == `[0,0,0,0,1,0,0]` if `maxEntries = 8` and "dog" is the 3rd most frequent words.
#### Read file data array
```ruby
@dict.readFileDataArray(fileDataArray [[]string], fileIndex, start, input_len, predict_len [integer]) -> [string], [string]
```
* fileDataArray is an array of strings.
* first sub-string is: fileDataArray[fileIndex][start, input_len]
* second sub-string is: fileDataArray[fileIndex][start+predict_len, input_len+predict_len]
#### Encode array 
```ruby
@dict.encodeArray(wordArray [string or []string], hash=@stringToEncoding) -> array [Numo:NArray]
```
* converts word array [strings] to encoded array via hash.
#### Decode array
```ruby
@dict.decodeArray(encodedArrray [Numo::NArray], hash=@encodingToString) -> output [string]
```
* If an array doesn't have a decoding, "@" is appened to the string.
#### Decode array by maximum 
```ruby
@dict.decodeArrayByMaximum(encodedArrray [Numo::NArray], hash=@encodingToString) -> output [string]
```
* If hot encoded array is not binary, max elment is considered the key index.
* `[0.3,0.1,0.2,0.7] == [0,0,0,1]`
#### Binary to array
```ruby
@dict.binaryToArray(binary, x_dim [integer]) -> binaryArray [Numo::NArray]
```
* converts input integer to binary array of fixed length `x_dim`
#### Array to binary 
```ruby
@dict.arrayToBinary(array [Numo::NArray]) -> binary [integer]
```
#### View hash
```ruby
@dict.viewHash([hash])
```

