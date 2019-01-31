# `repr_align_enum`

The tracking issue for this feature is: [#57996]

[#57996]: https://github.com/rust-lang/rust/issues/57996

------------------------

The `repr_align_enum` feature allows using the `#[repr(align(x))]` attribute
on enums, similarly to structs.

# Examples

```rust
#![feature(repr_align_enum)]

#[repr(align(8))]
enum Aligned {
    Foo,
    Bar { value: u32 },
}

fn main() {
    assert_eq!(std::mem::align_of::<Aligned>(), 8);
}
```

This is equivalent to using an aligned wrapper struct everywhere:

```rust
#[repr(align(8))]
struct Aligned(Unaligned);

enum Unaligned {
    Foo,
    Bar { value: u32 },
}

fn main() {
    assert_eq!(std::mem::align_of::<Aligned>(), 8);
}
```
