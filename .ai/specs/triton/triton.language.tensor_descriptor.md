Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
-   triton.language
    -   Programming Model
        -   triton.language.tensor
        -   triton.language.tensor_descriptor
            -   `tensor_descriptor`
        -   triton.language.program_id
        -   triton.language.num_programs
    -   Creation Ops
    -   Shape Manipulation Ops
    -   Linear Algebra Ops
    -   Memory/Pointer Ops
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
-   triton.language.tensor_descriptor
-   View page source

------------------------------------------------------------------------

# triton.language.tensor_descriptor¶

*class *triton.language.tensor_descriptor(*self*, *handle*, *shape: List\[tensor\]*, *strides: List\[tensor\]*, *block_type: block_type*)¶

:   A descriptor representing a tensor in global memory.

    \_\_init\_\_(*self*, *handle*, *shape: List\[tensor\]*, *strides: List\[tensor\]*, *block_type: block_type*)¶

    :   Not called by user code.

    Methods

      ---------------------------------------------------- --------------------------------------------------------------------------
      `__init__`(self, handle, shape, strides, \...)       Not called by user code.
      `atomic_add`(self, offsets, value\[, \_semantic\])   
      `atomic_and`(self, offsets, value\[, \_semantic\])   
      `atomic_max`(self, offsets, value\[, \_semantic\])   
      `atomic_min`(self, offsets, value\[, \_semantic\])   
      `atomic_or`(self, offsets, value\[, \_semantic\])    
      `atomic_xor`(self, offsets, value\[, \_semantic\])   
      `gather`(self, \*args\[, \_semantic\])               Gather multiple descriptors worth of data
      `load`(self, offsets\[, \_semantic\])                Load a block from the descriptor starting at the given element offsets.
      `scatter`(self, value, \*args\[, \_semantic\])       Scatter multiple descriptors worth of data
      `store`(self, offsets, value\[, \_semantic\])        Store a block from the descriptor starting at the given element offsets.
      ---------------------------------------------------- --------------------------------------------------------------------------

    Attributes

      --------------- --
      `block_shape`   
      `block_type`    
      `dtype`         
      `type`          
      --------------- --

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
