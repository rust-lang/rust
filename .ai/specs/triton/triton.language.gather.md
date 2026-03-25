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
    -   Math Ops
    -   Reduction Ops
    -   Scan/Sort Ops
        -   triton.language.associative_scan
        -   triton.language.cumprod
        -   triton.language.cumsum
        -   triton.language.histogram
        -   triton.language.sort
        -   triton.language.gather
            -   `gather()`
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
-   triton.language.gather
-   View page source

------------------------------------------------------------------------

# triton.language.gather¶

triton.language.gather(*src*, *index*, *axis*, *\_semantic=None*)¶

:   Gather from a tensor along a given dimension.

    Parameters:

    :   -   **src** (*Tensor*) -- the source tensor

        -   **index** (*Tensor*) -- the index tensor

        -   **axis** (*int*) -- the dimension to gather along

    This function can also be called as a member function on `tensor`, as `x.gather(...)` instead of `gather(x,`` ``...)`.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
