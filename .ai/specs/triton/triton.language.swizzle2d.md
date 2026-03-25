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
        -   triton.language.swizzle2d
            -   `swizzle2d()`
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
-   triton.language.swizzle2d
-   View page source

------------------------------------------------------------------------

# triton.language.swizzle2d¶

triton.language.swizzle2d(*i*, *j*, *size_i*, *size_j*, *size_g*)¶

:   Transforms the indices of a row-major size_i \* size_j matrix into the indices of a column-major matrix for each group of size_g rows.

    For example, for `size_i`` ``=`` ``size_j`` ``=`` ``4` and `size_g`` ``=`` ``2`, it will transform

        [[0 , 1 , 2 , 3 ],
         [4 , 5 , 6 , 7 ],
         [8 , 9 , 10, 11],
         [12, 13, 14, 15]]

    into

        [[0, 2,  4 , 6 ],
         [1, 3,  5 , 7 ],
         [8, 10, 12, 14],
         [9, 11, 13, 15]]

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
