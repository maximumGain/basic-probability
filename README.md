# basic-probability
Matlab source code for basic probability used in information theory.
Static methods are defined in a class named 'probabilityTool'. 
Simply call a function in this class by probabilityTool.<functionname>.
All methods provides default input for computation.   
Your inputs are collected using <varargin>.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Example:
Jointly distributed random variables X and Y are defined with a joint probability distribution pXY(x,y) = Pr(X = x, Y = y).
Then the marginal distribution pX(x) and pY(y) can be computed by:
[px,py] = probabilityTool.marginalize(pxy)          %pxy is the joint probability distribution pXY(x,y)
or
[px,py] = probabilityTool.marginalize(pxy,'r')       which returns rational number.

The program also provides default values, invoking 
[px,py] = probabilityTool.marginalize


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Type methods(probabilityTool) to see all methods in this class.

The software provides the following functions.
-   Theorem of total probability                                       @probabilityTool.PRy   
-   Marginalization                                                           @probabilityTool.marginalize 
-   Compute joint distribution                                         @probabilityTool.computeJointDistribution
-   Compute "all-related" probability                              @probabilityTool.computeAllP
-   Bayes rule                                                                  @probabilityTool.bayes
-   Compute expected value                                           @probabilityTool.computeExpection
-   Expected value of a function                                     @probabilityTool.expectionOfaFunction 
-   Compute variance                                                     @probabilityTool.computeVariance
-   An example of binary random vector                        @probabilityTool.binaryRandomVectorExample (uses @binomialPR)
-   Use Chebyshev inequality on random vectors          @largeNumberExperiment (uses @randomSamples)


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Notations in source code:
-   px:         Pr(X = x)
-   py:         Pr(Y = y)
-   pxy:       Pr(X=x,Y=y) joint probability
-   pxgy:     Pr(X=x | Y=y) conditional probability
-   pygx:     Pr(Y=y | X=x) conditional probability
-   EX:        E[X] expection of pX(x)
-   EgX:      E[g(x)] expection of a function g(x), where g(x) is a function of pX(x)
-   Var:       Var[X] variance of pX(x)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------




v1.0    June 29, 2020    Initial release
