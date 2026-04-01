# `asm_goto_with_outputs`

The tracking issue for this feature is: [#119364]

[#119364]: https://github.com/rust-lang/rust/issues/119364

------------------------

This feature allows label operands to be used together with output operands.

Example:
```rust,ignore (partial-example, x86-only)

unsafe {
    let a: usize;
    asm!(
        "mov {}, 1"
        "jmp {}",
        out(reg) a,
        label {
            println!("Jumped from asm {}!", a);
        }
    );
}
```

The output operands are assigned before the label blocks are executed.
