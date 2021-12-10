# `asm_const`

The tracking issue for this feature is: [#72016]

[#72016]: https://github.com/rust-lang/rust/issues/72016

------------------------

This feature adds a `const <expr>` operand type to `asm!` and `global_asm!`.
- `<expr>` must be an integer constant expression.
- The value of the expression is formatted as a string and substituted directly into the asm template string.
