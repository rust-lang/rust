# `indirect-branch-cs-prefix`

The tracking issue for this feature is: https://github.com/rust-lang/rust/issues/116852.

------------------------

Option `-Zindirect-branch-cs-prefix` controls whether a `cs` prefix is added to
`call` and `jmp` to indirect thunks.

It is equivalent to [Clang]'s and [GCC]'s `-mindirect-branch-cs-prefix`. The
Linux kernel uses it for RETPOLINE builds. For details, see
[LLVM commit 6f867f910283] ("[X86] Support ``-mindirect-branch-cs-prefix`` for
call and jmp to indirect thunk") which introduces the feature.

Only x86 and x86_64 are supported.

[Clang]: https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mindirect-branch-cs-prefix
[GCC]: https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#index-mindirect-branch-cs-prefix
[LLVM commit 6f867f910283]: https://github.com/llvm/llvm-project/commit/6f867f9102838ebe314c1f3661fdf95700386e5a
