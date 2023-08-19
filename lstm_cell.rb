#!/usr/bin/env ruby
require 'matrix' # https://www.rubyguides.com/2019/01/ruby-matrix/
require 'superators19' # https://stackoverflow.com/questions/11874579/define-custom-ruby-operator
require 'gnuplot'
require 'gnuplot/multiplot'
=begin 
	This is an LSTM cell programmed in RUBY.
=end
class Matrix
	superator "*~" do |operand|
		self.hadamard_product(operand)
	end
end
class LSTM_CELL
=begin
	This method initialises the matrices used in an LSTM Cell

	For a more clear understanding of what each variable does, 
	please see the fully annotated image on GitHub.

	In order for this implementation to work, input size to the 
	LSTM cell must equal the output size. ins = outs = "sx"
=end
	def init(sz, alpha)
		# Gate matrices
		# Z states = SUM(Wxg*~, Whg*~, bg)
		# meaning they are the same dimensions
		# as the bias vectors
		@Zg = Matrix.build(sz, 1) { 0 }
		@Zi = Matrix.build(sz, 1) { 0 }
		@Zf = Matrix.build(sz, 1) { 0 }
		@Zo = Matrix.build(sz, 1) { 0 }
		# Weight matrices (x)
		@Wxg = Matrix.identity(sz) *~ (Matrix.build(sz, sz) { rand }) # Weight matrices are filled with
		@Wxi = Matrix.identity(sz) *~ (Matrix.build(sz, sz) { rand }) # random values (in identity matrix)
		@Wxf = Matrix.identity(sz) *~ (Matrix.build(sz, sz) { rand })
		@Wxo = Matrix.identity(sz) *~ (Matrix.build(sz, sz) { rand })
		# Weight matrices (h) 
		@Whg = Matrix.identity(sz) *~ (Matrix.build(sz, sz) { rand })
		@Whi = Matrix.identity(sz) *~ (Matrix.build(sz, sz) { rand })
		@Whf = Matrix.identity(sz) *~ (Matrix.build(sz, sz) { rand })
		@Who = Matrix.identity(sz) *~ (Matrix.build(sz, sz) { rand })
		# Bias matrices 
		@Bg = Matrix.build(sz, 1) { rand } # bias vectors are applied to the output
		@Bi = Matrix.build(sz, 1) { rand } # hence they are len(out)
		@Bf = Matrix.build(sz, 1) { rand } # BIAS WAS "rand"
		# Gate vectors
		# Gate vectors are the tanh or sigoid 
		# outputs of the Z states, and hence
		# are the same dimension as the Z states
		@Gt = Matrix.build(sz, 1) { 0 }
		@It = Matrix.build(sz, 1) { 0 }
		@Ft = Matrix.build(sz, 1) { 0 }
		@Ot = Matrix.build(sz, 1) { 0 }
		# gates
		@Ogate = Matrix.build(sz, 1) { 0 }
		@Igate = Matrix.build(sz, 1) { 0 }
		@Fgate = Matrix.build(sz, 1) { 0 }
		# Cell states
		# Cell states are the only states that use
		# the input size (aside from 1 dimension of 
		# the weight matrices)
		@Ct = Matrix.build(sz, 1) { 0 }
		@Ht = Matrix.build(sz, 1) { 0 }
		# Previous cell states
		@Ctm1 = Matrix.build(sz, 1) { 1 } # defaults are overridden when setting the 
		@Htm1 = Matrix.build(sz, 1) { 0 } # LSTM cell state
		@Xtm1 = Matrix.build(sz, 1) { 1 }
		# Cell properties
		@sz = sz
		@@U = Matrix.build(1, sz) { 1 }
		@@Alpha = alpha
		@Yt = Matrix.build(sz, 1) { 0 }
	end
=begin 
	This method prints the current weights in a readable syntax.
	useful for debugging to see how gradient descent is improving training.

	An additional function (in future) would be to save the weights to a CSv file.
	That way changes in weight matrices could be visualised in a 3D plot using MATLAB.
=end
	def viewWeights()
		puts "Wxg"
		puts @Wxg.to_a.map(&:inspect)
		puts "Wxi"
		puts @Wxi.to_a.map(&:inspect)
		puts "Wxf"
		puts @Wxf.to_a.map(&:inspect)
		puts "Wxo"
		puts @Wxo.to_a.map(&:inspect)
		puts "Whg"
		puts @Whg.to_a.map(&:inspect)
		puts "Whi"
		puts @Whi.to_a.map(&:inspect)
		puts "Whf"
		puts @Whf.to_a.map(&:inspect)
		puts "Who"
		puts @Who.to_a.map(&:inspect)
	end
	def plotWeights()
		Gnuplot.open do |gp|
			Gnuplot::Multiplot.new(gp, layout: [2,4], title: "Weight matrices") do |mp|
			  	Gnuplot::SPlot.new(mp) do |plot|
			  		plot.title  "Wxg"
			  		plot.grid
					plot.pm3d
			   		plot.hidden3d
			    	plot.palette 'defined (   0 "black", 51 "blue", 102 "green", 153 "yellow", 204 "red", 255 "white" )'
				    plot.data << Gnuplot::DataSet.new( [(1..@sz), (1..@sz), @Wxg] ) do |ds|
				    	ds.with = "pm3d"
			      		ds.matrix = true
				    end
				end
				Gnuplot::SPlot.new(mp) do |plot|
					plot.title  "Wxi"
				   	plot.data << Gnuplot::DataSet.new( [(1..@sz), (1..@sz), @Wxi] ) do |ds|
			      		ds.with = "pm3d"
			      		ds.matrix = true
			      	end
			  	end
			  	Gnuplot::SPlot.new(mp) do |plot|
				  	plot.title  "Wxf"
				   	plot.data << Gnuplot::DataSet.new( [(1..@sz), (1..@sz), @Wxf] ) do |ds|
			      		ds.with = "pm3d"
			      		ds.matrix = true
			      	end
			  	end
			  	Gnuplot::SPlot.new(mp) do |plot|
				  	plot.title  "Wxo"
				   	plot.data << Gnuplot::DataSet.new( [(1..@sz), (1..@sz), @Wxo] ) do |ds|
			      		ds.with = "pm3d"
			      		ds.matrix = true
			      	end
			  	end
			  	Gnuplot::SPlot.new(mp) do |plot|
				    plot.title  "Whg"
					plot.data << Gnuplot::DataSet.new( [(1..@sz), (1..@sz), @Whg] ) do |ds|
				    	ds.with = "pm3d"
				    	ds.matrix = true
				    end
				end
				Gnuplot::SPlot.new(mp) do |plot|
				  	plot.title  "Whi"
				   	plot.data << Gnuplot::DataSet.new( [(1..@sz), (1..@sz), @Whi] ) do |ds|
			      		ds.with = "pm3d"
			      		ds.matrix = true
			      	end
			  	end
			  	Gnuplot::SPlot.new(mp) do |plot|
				  	plot.title  "Whf"
				    plot.data << Gnuplot::DataSet.new( [(1..@sz), (1..@sz), @Whf] ) do |ds|
			      		ds.with = "pm3d"
			      		ds.matrix = true
			      	end
			  	end
			  	Gnuplot::SPlot.new(mp) do |plot|
				  	plot.title  "Who"
				   	plot.data << Gnuplot::DataSet.new( [(1..@sz), (1..@sz), @Who] ) do |ds|
			      		ds.with = "pm3d"
			      		ds.matrix = true
			      	end
			  	end
			end
		end
	end
=begin
	This method prints the current state of the LSTM cell
	Ht and Ct
=end
	def viewState()
		puts "Ct"
		puts @Ct.to_a.map(&:inspect)
		puts "Ht"
		puts @Ht.to_a.map(&:inspect)
		puts "Yt"
		puts @Yt.to_a.map(&:inspect)
		puts "Xtm1"
		puts @Xtm1.to_a.map(&:inspect)
	end
=begin
	This sets the input state of the LSTM machine.
	Done before forward propagation.

	It also sets the output target @Yt
=end
	def setState(ctm1_, htm1_, xtm1_, yt_)
		@Ctm1 = ctm1_
		@Htm1 = htm1_
		@Xtm1 = xtm1_
		@Yt = yt_
	end
=begin
	This function implements the forward propagation algorithm of
	the LSTM cell.

	For more context, and the full equations, see GitHub.
=end
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
=begin
	This method implements gradient descent backward propagation.
	This method also updates weight and bais matrices after calculating
	gradient. 
=end
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
=begin 
	This method updates the weight matrix using the gradients.
	Itterate over diagonal matrix, append vector value to each diagonal value.
=end
	def updateWeights(matrix_, vector_, alpha)
		for index in 0...matrix_.column_count()
			matrix_[index, index] = matrix_[index, index] + (vector_[0, index] * alpha) # ADD OR MINUS?
		end
		return matrix_
	end
=begin
	testing
=end
	def multiplyWithWeights(matrix_, weight_)
		out = Matrix.build(1, weight_.row_count()) { 0 }
		for i in 0...weight_.row_count()
			row_sum = 0
			for j in 0...weight_.column_count()
				row_sum = row_sum + (matrix_[0,j] * weight_[i,j])
			end
			out[0, i] = row_sum
		end
		return out
	end
=begin 
	Performs sigmoid function on 1D matrix.

	following function performs sigmoid on 1 value.
=end
	def sigmoidVector(vector_)
		output_vector = vector_
		#input vector must be a horisontal vector
		for index in 0...vector_.column_count()
			output_vector[0, index] = sigmoid(vector_[0, index])
		end
		return output_vector
	end
	def sigmoid(value_)
		output_value =  1 / (1 + Math.exp(-1 * value_))
		return output_value
	end
	def invSigmoidVector(vector_)
		output_vector = vector_
		#input vector must be a horisontal vector
		for index in 0...vector_.column_count()
			output_vector[0, index] = 1 - sigmoid(vector_[0, index])
		end
		return output_vector
	end
=begin
	This method performs the tanh function on a 1D matrix. 
=end
	def tanhVector(vector_)
		output_vector = vector_
		#input vector must be a horisontal vector
		for index in 0...vector_.column_count()
			output_vector[0, index] = Math.tanh(vector_[0, index])
		end
		return output_vector
	end
	def invTanhVector(vector_)
		output_vector = vector_
		#input vector must be a horisontal vector
		for index in 0...vector_.column_count()
			output_vector[0, index] = 1 - (Math.tanh(vector_[0, index]) * Math.tanh(vector_[0, index]))
		end
		return output_vector
	end
=begin
	These are the accessor methods 
=end
	def getCt()
		return @Ct
	end
	def getHt()
		return @Ht
	end
end
