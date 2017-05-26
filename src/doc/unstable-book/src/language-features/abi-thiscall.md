# `abi_thiscall`

The tracking issue for this feature is: [#42202]

[#42202]: https://github.com/rust-lang/rust/issues/42202

------------------------

The MSVC ABI on x86 Windows uses the `thiscall` calling convention for C++
instance methods by default; it is identical to the usual (C) calling
convention on x86 Windows except that the first parameter of the method,
the `this` pointer, is passed in the ECX register.
