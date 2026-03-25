Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
-   triton.language
    -   Programming Model
    -   Creation Ops
    -   Shape Manipulation Ops
    -   Linear Algebra Ops
    -   Memory/Pointer Ops
    -   Indexing Ops
        -   triton.language.flip
        -   triton.language.where
            -   `where()`
        -   triton.language.swizzle2d
    -   Math Ops
    -   Reduction Ops
    -   Scan/Sort Ops
    -   Atomic Ops
    -   Random Number Generation
    -   Iterators
    -   Inline Assembly
    -   Compiler Hint Ops
    -   Debug Ops
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
-   triton.language
-   triton.language.where
-   View page source

------------------------------------------------------------------------

# triton.language.where¶

triton.language.where(*condition*, *x*, *y*, *\_semantic=None*)¶

:   Returns a tensor of elements from either `x` or `y`, depending on `condition`.

    Note that `x` and `y` are always evaluated regardless of the value of `condition`.

    If you want to avoid unintended memory operations, use the `mask` arguments in triton.load and triton.store instead.

    The shape of `x` and `y` are both broadcast to the shape of `condition`. `x` and `y` must have the same data type.

    Parameters:

    :   -   **condition** (*Block* *of* *triton.bool*) -- When True (nonzero), yield x, otherwise yield y.

        -   **x** -- values selected at indices where condition is True.

        -   **y** -- values selected at indices where condition is False.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
