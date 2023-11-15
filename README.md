# mdp

Write a generic Markov process solver.


## *Running the Program*:
   - Open your command line interface (CLI).
   - Navigate to the directory containing the `main.py` file.
   - Run the program using the following command:
     ```
     python3 main.py [options] inputfile.txt
     ```
   - Replace `inputfile.txt` with the name of your test input file.

## *Command-Line Options*:
   - `-df`: Sets the discount factor. Default is `1.0`.
   - `-tol`: Sets the tolerance for convergence. Default is `0.001`.
   - `-iter`: Sets the maximum number of iterations. Default is `100`.
   - `-min`: Minimizes values as costs. If not provided, the program maximizes values as rewards.

## References
-coding guide from https://docs.python.org/3/
-some of the structure was inspired by  [tomasort's GitHub repository](https://github.com/tomasort).
