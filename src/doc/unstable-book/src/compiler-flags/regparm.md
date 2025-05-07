# `regparm`

The tracking issue for this feature is: https://github.com/rust-lang/rust/issues/131749.

------------------------

Option -Zregparm=N causes the compiler to pass N arguments
in registers EAX, EDX, and ECX instead of on the stack for "C", "cdecl", and "stdcall" fn.
It is UNSOUND to link together crates that use different values for this flag.
It is only supported on `x86`.

It is equivalent to [Clang]'s and [GCC]'s `-mregparm`.

Supported values for this option are 0-3.

[Clang]: https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mregparm
[GCC]: https://gcc.gnu.org/onlinedocs/gcc/x86-Function-Attributes.html#index-regparm-function-attribute_002c-x86

Implementation details:
For eligible arguments, llvm `inreg` attribute is set.
