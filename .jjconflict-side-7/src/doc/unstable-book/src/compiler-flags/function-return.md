# `function-return`

The tracking issue for this feature is: https://github.com/rust-lang/rust/issues/116853.

------------------------

Option `-Zfunction-return` controls how function returns are converted.

It is equivalent to [Clang]'s and [GCC]'s `-mfunction-return`. The Linux kernel
uses it for RETHUNK builds. For details, see [LLVM commit 2240d72f15f3] ("[X86]
initial -mfunction-return=thunk-extern support") which introduces the feature.

Supported values for this option are:

  - `keep`: do not convert function returns.
  - `thunk-extern`: convert function returns (`ret`) to jumps (`jmp`)
    to an external symbol called `__x86_return_thunk`.

Like in Clang, GCC's values `thunk` and `thunk-inline` are not supported.

Only x86 and non-large code models are supported.

[Clang]: https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mfunction-return
[GCC]: https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#index-mfunction-return
[LLVM commit 2240d72f15f3]: https://github.com/llvm/llvm-project/commit/2240d72f15f3b7b9d9fb65450f9bf635fd310f6f
