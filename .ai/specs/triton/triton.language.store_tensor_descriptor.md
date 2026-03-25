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
        -   triton.language.store_tensor_descriptor
            -   `store_tensor_descriptor()`
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
-   triton.language.store_tensor_descriptor
-   View page source

------------------------------------------------------------------------

# triton.language.store_tensor_descriptor¶

triton.language.store_tensor_descriptor(*desc: tensor_descriptor_base*, *offsets: Sequence\[constexpr \| tensor\]*, *value: tensor*, *\_semantic=None*) → tensor¶

:   Store a block of data to a tensor descriptor.

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
