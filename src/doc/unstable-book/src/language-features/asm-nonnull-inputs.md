# `asm_nonnull_inputs`

The tracking issue for this feature is: [#166132]

[#166132]: https://github.com/rust-lang/rust/issues/166132

------------------------

This feature allows thin `NonNull` pointers to be used as input
operands in inline assembly.

Example:
```rust,ignore (partial-example, x86-only)
let mut x = 1;
let p = core::ptr::NonNull::from_mut(&mut x);

unsafe {
    core::arch::asm!(
        "mov dword ptr [{}], 2",
        in(reg) p,
    )
};

assert_eq!(x, 2);
```
