# <img src="https://enzyme.mit.edu/logo.svg" width="75" align=left> The Enzyme High-Performance Automatic Differentiator of LLVM


Enzyme is a plugin that performs automatic differentiation (AD) of statically analyzable LLVM. It is highly-efficient and its ability perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.

Enzyme can be used by calling `__enzyme_autodiff` on a function to be differentiated as shown below:
```c
double foo(double);

double grad_foo(double x) {
    return __enzyme_autodiff(foo, x);
}
```

Running the Enzyme transformation pass then replaces the call to `__enzyme_autodiff` with the gradient of its first argument.

Information on installing and using Enzyme can be found on our website: [https://enzyme.mit.edu](https://enzyme.mit.edu).

To get involved or if you have questions, please join our [mailing list](https://groups.google.com/d/forum/enzyme-dev).

If using this code in an academic setting, please cite the following:
```
@misc{enzymeGithub,
 author = {William S. Moses and Valentin Churavy},
 title = {Enzyme: High Performance Automatic Differentiation of LLVM},
 year = {2020},
 howpublished = {\url{https://github.com/wsmoses/Enzyme}},
 note = {commit xxxxxxx}
}
```
