# `asm_sym`

The tracking issue for this feature is: [#72016]

[#72016]: https://github.com/rust-lang/rust/issues/72016

------------------------

This feature adds a `sym <path>` operand type to `asm!` and `global_asm!`.
- `<path>` must refer to a `fn` or `static`.
- A mangled symbol name referring to the item is substituted into the asm template string.
- The substituted string does not include any modifiers (e.g. GOT, PLT, relocations, etc).
- `<path>` is allowed to point to a `#[thread_local]` static, in which case the asm code can combine the symbol with relocations (e.g. `@plt`, `@TPOFF`) to read from thread-local data.
