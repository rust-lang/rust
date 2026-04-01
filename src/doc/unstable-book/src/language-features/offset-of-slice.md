# `offset_of_slice`

The tracking issue for this feature is: [#126151]

[#126151]: https://github.com/rust-lang/rust/issues/126151

------------------------

When the `offset_of_slice` feature is enabled, the [`offset_of!`] macro may be used to determine
the offset of fields whose type is `[T]`, that is, a slice of dynamic size.

In general, fields whose type is dynamically sized do not have statically known offsets because
they do not have statically known alignments. However, `[T]` has the same alignment as `T`, so
it specifically may be allowed.

```rust
#![feature(offset_of_slice)]

#[repr(C)]
pub struct Struct {
    head: u32,
    tail: [u8],
}

fn main() {
    assert_eq!(std::mem::offset_of!(Struct, tail), 4);
}
```

[`offset_of!`]: ../../std/mem/macro.offset_of.html
