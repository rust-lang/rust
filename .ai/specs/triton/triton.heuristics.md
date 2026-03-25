Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
    -   triton.jit
    -   triton.autotune
    -   triton.heuristics
        -   `heuristics()`
    -   triton.Config
-   triton.language
-   triton.testing
-   Triton Semantics

Triton MLIR Dialects

-   Triton MLIR Dialects and Ops

Programming Guide

-   Introduction
-   Related Work
-   Debugging Triton

Triton

-   
-   triton
-   triton.heuristics
-   View page source

------------------------------------------------------------------------

# triton.heuristics¶

triton.heuristics(*values*)¶

:   Decorator for specifying how the values of certain meta-parameters may be computed. This is useful for cases where auto-tuning is prohibitively expensive, or just not applicable.

        # smallest power-of-two >= x_size
        @triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
        @triton.jit
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...

    Parameters:

    :   **values** (*dict\[str,* *Callable\[\[dict\[str,* *Any\]\],* *Any\]\]*) -- a dictionary of meta-parameter names and functions that compute the value of the meta-parameter. each such function takes a list of positional arguments as input.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
