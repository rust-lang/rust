Triton

Getting Started

-   Installation
-   Tutorials

Python API

-   triton
-   triton.language
    -   Programming Model
        -   triton.language.tensor
            -   `tensor`
        -   triton.language.tensor_descriptor
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
-   triton.language.tensor
-   View page source

------------------------------------------------------------------------

# triton.language.tensor¶

*class *triton.language.tensor(*self*, *handle*, *type: dtype*)¶

:   Represents an N-dimensional array of values or pointers.

    `tensor` is the fundamental data structure in Triton programs. Most functions in `triton.language` operate on and return tensors.

    Most of the named member functions here are duplicates of the free functions in `triton.language`. For example, `triton.language.sqrt(x)` is equivalent to `x.sqrt()`.

    `tensor` also defines most of the magic/dunder methods, so you can write `x+y`, `x`` ``<<`` ``2`, etc.

    Constructors

    \_\_init\_\_(*self*, *handle*, *type: dtype*)¶

    :   Not called by user code.

    Methods

      --------------------------------------------------------- -------------------------------------------------------------------------------------------
      `__init__`(self, handle, type)                            Not called by user code.
      `abs`(self\[, \_semantic\])                               Forwards to `abs()` free function
      `advance`(self, offsets\[, \_semantic\])                  Forwards to `advance()` free function
      `argmax`(input, axis\[, tie_break_left, keep_dims\])      Returns the maximum index of all elements in the `input` tensor along the provided `axis`
      `argmin`(input, axis\[, tie_break_left, keep_dims\])      Returns the minimum index of all elements in the `input` tensor along the provided `axis`
      `associative_scan`(self, axis, combine_fn\[, \...\])      Forwards to `associative_scan()` free function
      `atomic_add`(self, val\[, mask, sem, scope, \...\])       Forwards to `atomic_add()` free function
      `atomic_and`(self, val\[, mask, sem, scope, \...\])       Forwards to `atomic_and()` free function
      `atomic_cas`(self, cmp, val\[, sem, scope, \...\])        Forwards to `atomic_cas()` free function
      `atomic_max`(self, val\[, mask, sem, scope, \...\])       Forwards to `atomic_max()` free function
      `atomic_min`(self, val\[, mask, sem, scope, \...\])       Forwards to `atomic_min()` free function
      `atomic_or`(self, val\[, mask, sem, scope, \...\])        Forwards to `atomic_or()` free function
      `atomic_xchg`(self, val\[, mask, sem, scope, \...\])      Forwards to `atomic_xchg()` free function
      `atomic_xor`(self, val\[, mask, sem, scope, \...\])       Forwards to `atomic_xor()` free function
      `broadcast_to`(self, \*shape\[, \_semantic\])             Forwards to `broadcast_to()` free function
      `cast`(self, dtype\[, fp_downcast_rounding, \...\])       Forwards to `cast()` free function
      `cdiv`(x, div)                                            Computes the ceiling division of `x` by `div`
      `ceil`(self\[, \_semantic\])                              Forwards to `ceil()` free function
      `cos`(self\[, \_semantic\])                               Forwards to `cos()` free function
      `cumprod`(input\[, axis, reverse\])                       Returns the cumprod of all elements in the `input` tensor along the provided `axis`
      `cumsum`(input\[, axis, reverse, dtype\])                 Returns the cumsum of all elements in the `input` tensor along the provided `axis`
      `erf`(self\[, \_semantic\])                               Forwards to `erf()` free function
      `exp`(self\[, \_semantic\])                               Forwards to `exp()` free function
      `exp2`(self\[, \_semantic\])                              Forwards to `exp2()` free function
      `expand_dims`(self, axis\[, \_semantic\])                 Forwards to `expand_dims()` free function
      `flip`(x\[, dim\])                                        Flips a tensor x along the dimension dim.
      `floor`(self\[, \_semantic\])                             Forwards to `floor()` free function
      `gather`(self, index, axis\[, \_semantic\])               Forwards to `gather()` free function
      `histogram`(self, num_bins\[, mask, \_semantic, \...\])   Forwards to `histogram()` free function
      `item`(self\[, \_semantic, \_generator\])                 Forwards to `item()` free function
      `log`(self\[, \_semantic\])                               Forwards to `log()` free function
      `log2`(self\[, \_semantic\])                              Forwards to `log2()` free function
      `logical_and`(self, other\[, \_semantic\])                
      `logical_or`(self, other\[, \_semantic\])                 
      `max`(input\[, axis, return_indices, \...\])              Returns the maximum of all elements in the `input` tensor along the provided `axis`
      `min`(input\[, axis, return_indices, \...\])              Returns the minimum of all elements in the `input` tensor along the provided `axis`
      `permute`(self, \*dims\[, \_semantic\])                   Forwards to `permute()` free function
      `ravel`(x\[, can_reorder\])                               Returns a contiguous flattened view of `x`.
      `reduce`(self, axis, combine_fn\[, keep_dims, \...\])     Forwards to `reduce()` free function
      `reduce_or`(input, axis\[, keep_dims\])                   Returns the reduce_or of all elements in the `input` tensor along the provided `axis`
      `reshape`(self, \*shape\[, can_reorder, \...\])           Forwards to `reshape()` free function
      `rsqrt`(self\[, \_semantic\])                             Forwards to `rsqrt()` free function
      `sigmoid`(x)                                              Computes the element-wise sigmoid of `x`.
      `sin`(self\[, \_semantic\])                               Forwards to `sin()` free function
      `softmax`(x\[, dim, keep_dims, ieee_rounding\])           Computes the element-wise softmax of `x`.
      `sort`(self\[, dim, descending\])                         
      `split`(self\[, \_semantic, \_generator\])                Forwards to `split()` free function
      `sqrt`(self\[, \_semantic\])                              Forwards to `sqrt()` free function
      `sqrt_rn`(self\[, \_semantic\])                           Forwards to `sqrt_rn()` free function
      `store`(self, value\[, mask, boundary_check, \...\])      Forwards to `store()` free function
      `sum`(input\[, axis, keep_dims, dtype\])                  Returns the sum of all elements in the `input` tensor along the provided `axis`
      `to`(self, dtype\[, fp_downcast_rounding, \...\])         Alias for `tensor.cast()`.
      `trans`(self, \*dims\[, \_semantic\])                     Forwards to `trans()` free function
      `view`(self, \*shape\[, \_semantic\])                     Forwards to `view()` free function
      `xor_sum`(input\[, axis, keep_dims\])                     Returns the xor sum of all elements in the `input` tensor along the provided `axis`
      --------------------------------------------------------- -------------------------------------------------------------------------------------------

    Attributes

      -------- -------------------------
      `T`      Transposes a 2D tensor.
      `type`   
      -------- -------------------------

Previous Next

------------------------------------------------------------------------

© Copyright 2020, Philippe Tillet.

Built with Sphinx using a theme provided by Read the Docs.
