# `reg-struct-return`

The tracking issue for this feature is: https://github.com/rust-lang/rust/issues/116973.

------------------------

Option -Zreg-struct-return causes the compiler to return small structs in registers
instead of on the stack for extern "C"-like functions.
It is UNSOUND to link together crates that use different values for this flag.
It is only supported on `x86`.

It is equivalent to [Clang]'s and [GCC]'s `-freg-struct-return`.

[Clang]: https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-freg-struct-return
[GCC]: https://gcc.gnu.org/onlinedocs/gcc/Code-Gen-Options.html#index-freg-struct-return
