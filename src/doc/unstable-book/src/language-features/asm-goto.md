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

The block must have unit type or diverge.

When `label <block>` is used together with `noreturn` option, it means that the
assembly will not fallthrough. It's allowed to jump to a label within the
assembly. In this case, the entire `asm!` expression will have an unit type as
opposed to diverging, if not all label blocks diverge. The `asm!` expression
still diverges if `noreturn` option is used and all label blocks diverge.
