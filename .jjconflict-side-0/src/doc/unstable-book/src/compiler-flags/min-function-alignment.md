# `min-function-alignment`

The tracking issue for this feature is: https://github.com/rust-lang/rust/issues/82232.

------------------------

The `-Zmin-function-alignment=<align>` flag specifies the minimum alignment of functions for which code is generated.
The `align` value must be a power of 2, other values are rejected.

Note that `-Zbuild-std` (or similar) is required to apply this minimum alignment to standard library functions.
By default, these functions come precompiled and their alignments won't respect the `min-function-alignment` flag.

This flag is equivalent to:

- `-fmin-function-alignment` for [GCC](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fmin-function-alignment_003dn)
- `-falign-functions` for [Clang](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang1-falign-functions)

The specified alignment is a minimum. A higher alignment can be specified for specific functions by using the [`align(...)`](https://github.com/rust-lang/rust/issues/82232) feature and annotating the function with a `#[align(<align>)]` attribute. The attribute's value is ignored when it is lower than the value passed to `min-function-alignment`.

There are two additional edge cases for this flag:

- targets have a minimum alignment for functions (e.g. on x86_64 the lowest that LLVM generates is 16 bytes).
    A `min-function-alignment` value lower than the target's minimum has no effect.
- the maximum alignment supported by rust (and LLVM) is `2^29`. Trying to set a higher value results in an error.
