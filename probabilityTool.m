classdef probabilityTool
    
    methods(Static)
        
        %--------------------------------1
        %Jointly distributed random variables X and Y are defined for sample spaces X and Y respectively,
        %with a joint probability distribution: pX,Y(x, y) = Pr(X = x, Y = y)
        %The joint distribution pX,Y (x, y) is the “all-knowing distribution.”
        %This means that given pX,Y(x,y), it is possible to compute pX(x), pY(y), pX|Y(x|y) and pY|X(y|x).
        %If you are given only pY|X(y|x), then you additionally need pX(x) to find
        %the joint distribution.
        %--------------------------------1
        function py = PRy(varargin)
            %"Theorem of total probability" PX(x) = sum_y pY(y)PX|Y(x|y)
            %compute PY(y) = sum_x pX(x)PY|X(y|x)
            %pygx is "probability of Y given X", as a matrix of size(X) rows and size(Y) columns. Rows sum to 1.
            %
            %Call this function by: py = probabilityTool.PRy(pygx,px)
            %If nargin = 0, use default pygx and px.
            assert(nargin <= 2,'Too many input arguments');
            assert(nargin ~= 1,'Not enough input arguments');
            if isempty(varargin)
                pygx = [...
                    0     1     0;
                    1/2     0   1/2;
                    1/3   1/3   1/3;
                    0     2/3   1/3];
                px = [1/4 3/8 1/8 1/4];
            else
                pygx = varargin{1};
                px = varargin{2};
            end
            assert(length(pygx)>1 && length(px)>1,'Invalid input')
            assert(all(pygx(:) >= 0),'pygx are not all positive')
            assert(all(px >= 0),'px are not all positive')
            assert(all(abs(1-sum(pygx,2))<1E-10),'Each row of pygx should sum to 1');
            assert(abs(1-sum(px))<1E-10,'px does not sum to 1');
            py = px * pygx;
        end
        
        function [px,py] = marginalize(varargin)
            %"Marginalization" PY(y) = sum_y PX,Y(x,y) and PX(x) = sum_x PX,Y(x,y)
            %Call this function by: [px,py] = probabilityTool.marginalize(pxy)
            %Return rational number px and py by: [px,py] = probabilityTool.marginalize(pxy,'r')
            %If nargin = 0, use default pxy.
            assert(nargin <= 2,'Too many input arguments');
            if isempty(varargin)
                pxy = [1/2 0 0; 1/4 1/16 0; 0 1/8 1/16];
            else
                pxy = varargin{1};
            end
            assert(length(pxy)> 1,'Invalid pxy')
            assert(all(pxy(:) >= 0),'pxy are not all positive')
            assert(abs(1-sum(pxy,'all'))<1E-10,'px does not sum to 1');
            py = sum(pxy,1);
            px = sum(pxy,2);
            if nargin == 2%return rational numbers
                if strcmp(varargin{2},'r')
                    
                    px = rats(px);
                    py = rats(py);
                else
                    fprintf('Invalid second input argument.')
                end
            end
        end
        
        function pxy = computeJointDistribution(varargin)
            %"Compute joint distribution" PX,Y(x,y) = PY|X(y,x)PX(x)
            %pxy = probabilityTool.computeJointDistribution(pygx,px)
            %Empty varargin uses default pygx and px
            assert(nargin <= 2,'Too many input arguments');
            assert(nargin ~= 1,'Not enough input arguments');
            if isempty(varargin)
                pygx = [1 0 0 ; 4/5 1/5 0 ; 0 2/3 1/3];
                px = [1/2 1/4 1/4];
            else
                pygx = varargin{1};
                px = varargin{2};
            end
            assert(length(pygx)> 1,'Invalid pygx')
            assert(all(pygx(:) >= 0),'pygx are not all positive')
            assert(all(px >= 0),'px are not all positive')
            assert(all(abs(1-sum(pygx,2))<1E-10),'Each row of pygx should sum to 1');
            assert(abs(1-sum(px))<1E-10,'px does not sum to 1');
            
            [~,Y] = size(pygx);
            pxy = repmat(px(:),1,Y) .* pygx;
        end
        
        function [px,py,pxgy,pygx] = computeAllP(varargin)
            %Given a joint distribution pX,Y(x,y), we can compute px,py,pxgy and pygx
            %[px,py,pxgy,pygx] = computeAllP(pxy)
            %[px,py,pxgy,pygx] = computeAllP(pxy,'r') returns rational numbers
            %Empty varargin uses default pxy
            assert(nargin <= 2,'Too many input arguments');
            if isempty(varargin)
                pxy=[1/2 0 0;1/4 1/16 0;0 1/8 1/16];
            else
                pxy = varargin{1};
            end
            assert(length(pxy)> 1,'Invalid pxy')
            assert(all(pxy(:) >= 0),'pxy are not all positive')
            assert(abs(1-sum(pxy,'all'))<1E-10,'px does not sum to 1');
            [X,Y] = size(pxy);
            py = sum(pxy,1);
            px = sum(pxy,2);
            %"Compute the conditional probability" PX|Y(x|y) = PX,Y(x,y)/PY(y)
            pxgy = pxy' ./ repmat(py(:)',X,1)';%transpose: rows sum to 1
            %pxgy is a matrix of Y rows and X columns.
            %"Compute the conditional probability" PY|X(y|x) = PX,Y(x,y)/PX(x)
            pygx = pxy ./ repmat(px(:),1,Y);
            
            if nargin == 2%return rational numbers
                if strcmp(varargin{2},'r')
                    px = rats(px);
                    py = rats(py);
                    pxgy = rats(pxgy);
                    pygx = rats(pygx);
                else
                    fprintf('Invalid second input argument.')
                end
            end
        end
        
        function pxgy = bayes(varargin)
            %pxgy = probabilityTool.bayes(pygx,px,py)
            %Empty varargin uses default pygx, px, and py.
            %"Bayes rule" PX|Y(x|y) = PY|X(y|x)PX(x)/PY(y)
            assert(nargin <= 3,'Too many input arguments');
            if isempty(varargin)
                pygx = [1 0 0; 4/5 1/5 0; 0 2/3 1/3];
                px = [1/2 5/16 3/16];
                py = [3/4 3/16 1/16];
            else
                pygx = varargin{1};
                px = varargin{2};
                py = varargin{3};
            end
            assert(length(pygx)>1 && length(px)>1 && length(py)>1,'Invalid input')
            assert(all(pygx(:)>=0) && all(px>=0) && all(py>=0),'Input arguments are not all positive')
            assert(all(abs(1-sum(pygx,2))<1E-10),'Each row of pygx should sum to 1');
            assert(abs(1-sum(px))<1E-10,'px does not sum to 1');
            assert(abs(1-sum(py))<1E-10,'px does not sum to 1');
            
            [X,Y] = size(pygx);
            pxgy = transpose(pygx .* repmat(px(:),1,Y) ./ repmat(py(:)',X,1));
        end
        
        
        
        
        %--------------------------------2
        %Two random variables X and Y are independent if and only if: pX,Y(x,y) = pX(x)pY(y)
        %Also, X and Y are independent if an only if pX|Y(x|y) = pX(x) for all x ∈ X,y ∈ Y.
        %If pX(x) = pY(x) for all x ∈ X, then we say X and Y are independent and identically distributed, often abbreviated iid.
        %--------------------------------2
        
        
        %--------------------------------3
        %Expectation and Variance
        %--------------------------------3
        
        function EX = computeExpection(varargin)
            %"Compute expected value" E[X] = sum_x x pX(x)
            %EX = probabilityTool.computeExpection(x,px)
            assert(nargin <= 2,'Too many input arguments');
            if isempty(varargin)
                %die roll random variable X
                x = 1:6; %\mathcal X = {1,2,3,4,5,6}
                px = repmat(1/6,6,1); %pX(x) = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
            else
                x = varargin{1};
                px = varargin{2};
            end
            EX = x*px(:);
        end
        
        function EgX = expectionOfaFunction(varargin)
            %"Compute expected value of a function" g(x)] E[g(X)] = sum_x g(x)pX(x)
            %EgX = probabilityTool.expectionOfaFunction(x,px,g)
            assert(nargin <= 3,'Too many input arguments');
            if isempty(varargin)
                x = [0 1 2 3];
                px = [1/4 3/8 1/8 1/4];
                g = @(x) x.^2; %g(x) = x^2
                %g = @(px) -log2(px); %g(x) = -log pX(x)
            else
                x = varargin{1};
                px = varargin{2};
                g = varargin{3};
            end
            EgX = g(x) * px(:);
        end
        
        function Var = computeVariance(varargin)
            %"compute variance" Var[X] = E[X^2] - (E[X])^2
            %Var = probabilityTool.computeVariance(x,px)
            assert(nargin <= 2,'Too many input arguments');
            if isempty(varargin)
                %die roll random variable X
                x = 1:6; %X = {1,2,3,4,5,6}
                px = repmat(1/6,6,1); %pX(x) = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
            else
                assert(isequal(length(varargin{1}),length(varargin{2})),'The two input arguments must have same length')
                x = varargin{1};
                px = varargin{2};
            end
            
            EX = x*px(:);
            g = @(x) x.^2; %g(x) = x^2
            EgX = g(x) * px(:);
            Var = EgX - (EX)^2;
        end
        
        %"The conditional expectation" E[X|Y = y] = sum_x xPX|Y(x|y)
        %The conditional expectation E[X|Y = y] is a function of y, for example, f(y) = E[X|Y = y].
        %Note that E[X|Y] is distinct from E[X|Y = y].
        %In particular E[X|Y] is a random variable equal to f(Y).
        %Since E[X|Y] is a random variable, it has an expectation. In fact, it is equal to E[X].
        %Proposition: [Law of Total Expectation] Let X and Y be jointly distributed random variables
        %E[E[X|Y]] = E[X]
        
        
        %"Expectation of Sums of Random Variables"
        %For any X1,X2,...,Xn and constants a1, a2,...,an, E[a1X1 + a2X2 + · · · anXn] = a1E[X1] + a2E[X2] + · · · anE[Xn]
        
        %"Variance of Sums of Random Variables"
        %For any independent X1,X2,...,Xn and constants a1, a2,...,an, Var[a1X1 + a2X2 + · · · anXn] = a21Var[X1] + a2Var[X2] + · · · a2nVar[Xn]
        
        
        
        %--------------------------------4
        %Random Vectors
        %--------------------------------4
        %Let X = X1,X2,...,Xn be a random vector of n random variables, independent and identically distributed.
        %A random variable Xi has distribution pX(x).
        %Then, the random vector X =(X1,X2,X3,...,Xn)
        
        %"The sample mean Xnbar" Xnbar = (1/n) sum Xi for i = 1,2,...,n
        %"The expected mean of the binary random vector" E[Xn_bar] = E[Xi]
        
        %"Binary random vector" If Xi is a binary random variable on sample space {0, 1},
        %with probability of a one equal to 0 ≤ p ≤ 1, then we say X is a binary random vector.
        %The expected mean of binary random vector X is E[Xn_bar] = E[Xi] = p.
        %The variance of binary random vector X is Var[Xn_bar] = p(1-p)/n.
        
        %The binary random vector X has k ones and n − k zeros.
        %Let K be the sum of X: K = sum Xi for i = 1,2,...,n, so that K is a random
        %variable expressing the number of ones in X. This is the binomial random
        %variable K with probability distribution
        %pK(k) = (n!/k!(n-k)!)p^k (1-p)^(n-k) for k = 0,1,...,n
        
        
        function [pX0100,pk,pk1or2] = binaryRandomVectorExample
            %Example: consider an example of a binary random vector with n = 4 and p = 1/4 (pX(1) = p = 1/4 and pX(0) = 1-p = 3/4).
            %The sample space is: {0000,0001,0010,...,1111}.
            %The probability of X = {0100} is pX0100 = pX(0)pX(1)pX(0)pX(1) by dependency.
            p = 1/4;
            pX0100 = (1-p)*p*(1-p)*(1-p);%27/256
            %The probability that X is a sequence of two 1's and two 0's: 27/128
            pk = binomialPR(4,2,p);
            %The probability that X has one 1 or X has two 1's: 81/128
            pk1 = binomialPR(4,1,p);
            pk2 = binomialPR(4,2,p);
            pk1or2 = pk1+pk2;
        end
        %--------------------------------5
        %Large Number Sample Mean Experiments
        %--------------------------------5
        
        function [probability, numberWithinEpsilon] = largeNumberExperiment(varargin)
            %Conduct a large number of experiments to find the number of
            %samples n within epsilon ε of the expected mean.
            %Then use this to compute the probability of being within ε of the expected mean.
            %The lower bound q: Pr(|Xn_bar-E[X]|<=ε)>=q, will be found using the Chebyshev inequality.
            %
            %[probability, numberWithinEpsilon] = probabilityTool.largeNumberExperiment(px,epsilon,n,numberOfExperiments)
            %
            %probability:q
            %sampleSpace:Xn_bar
            %trueMean:E[X]
            
            if isempty(varargin)
                px = [1/4 1/2 1/4];
                epsilon  = 0.1;
                n        = 50;
                numberOfExperiments = 10000;
            elseif nargin == 1
                px = varargin{1};
                epsilon  = 0.1;
                n        = 50;
                numberOfExperiments = 10000;
            elseif nargin == 2
                px = varargin{1};
                epsilon = varargin{2};
                n        = 50;
                numberOfExperiments = 10000;
            elseif nargin == 3
                px = varargin{1};
                epsilon = varargin{2};
                n        = varargin{3};
                numberOfExperiments = 10000;
            elseif nargin == 4
                px = varargin{1};
                epsilon = varargin{2};
                n        = varargin{3};
                numberOfExperiments = varargin{4};
            end
            assert(length(px) > 1 && abs(1-sum(px))<1E-10 && all(px >= 0),'Invalid px');
            
            sampleSpace = 1:length(px);
            trueMean = sampleSpace * px(:);%true mean of the random variable
            sampleMean = zeros(numberOfExperiments,1);
            for ii = 1:numberOfExperiments
                x = randomSamples(px,n);
                sampleMean(ii) = mean(x);
            end
            numberWithinEpsilon = length(find( abs(sampleMean - trueMean) < epsilon));
            probability = numberWithinEpsilon / numberOfExperiments;
        end
        
        
        
    end
end


function pk = binomialPR(n,k,p)
%pk = probabilityTool.binomialPR(n,k,p)
%"The General Binomial Probability Formula" P(k out of n) = pK(k) = (n!/k!(n-k)!) p^k (1-p)^(n-k)
%Great explanation at https://www.mathsisfun.com/data/binomial-distribution.html
pk = nchoosek(n,k) * p^k * (1-p)^(n-k);
end

function X = randomSamples(varargin)
%Generate n samples of a random variable according to a distribution pX(x)
%X = probabilityTool.randomSamples(px,n)
%Sample space = {1,2,3,...,length(px)}
if nargin == 0
    %Generate n = 10 samples from the distribution pX(x) = [1, 1, 1]
    px = [1/4 1/2 1/4];
    n = 10;
elseif nargin == 1
    px = varargin;
    n = 1; %default n
else
    px = varargin{1};
    n = varargin{2};
end

Fx = cumsum([0 px(1:end-1)]); %cumulative distribution
X = zeros(1,n); %pre-allocate
for ii = 1:n
    X(ii) = find(Fx < rand,1,'last'); %generate sample
end
end
