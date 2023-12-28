# `asm_goto`

The tracking issue for this feature is: [#119364]

[#119364]: https://github.com/rust-lang/rust/issues/119364

------------------------

This feature adds a `label <block>` operand type to `asm!`.

Example:
```rust,ignore (partial-example, x86-only)

unsafe {
    asm!(
        "jmp {}",
        label {
            println!("Jumped from asm!");
        }
    );
}
```

The block must have unit type.
