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
        -   triton.language.load
        -   triton.language.store
        -   triton.language.make_tensor_descriptor
        -   triton.language.load_tensor_descriptor
            -   `load_tensor_descriptor()`
        -   triton.language.store_tensor_descriptor
        -   triton.language.make_block_ptr
        -   triton.language.advance
    -   Indexing Ops
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
-   triton.language.load_tensor_descriptor
-   View page source

------------------------------------------------------------------------

# triton.language.load_tensor_descriptor¶

triton.language.load_tensor_descriptor(*desc: tensor_descriptor_base*, *offsets: Sequence\[constexpr \| tensor\]*, *\_semantic=None*) → tensor¶

:   Load a block of data from a tensor descriptor.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
