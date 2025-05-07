# `offset_of_enum`

The tracking issue for this feature is: [#120141]

[#120141]: https://github.com/rust-lang/rust/issues/120141

------------------------

When the `offset_of_enum` feature is enabled, the [`offset_of!`] macro may be used to obtain the
offsets of fields of `enum`s; to express this, `enum` variants may be traversed as if they were
fields. Variants themselves do not have an offset, so they cannot appear as the last path component.

```rust
#![feature(offset_of_enum)]
use std::mem;

#[repr(u8)]
enum Enum {
    A(u8, u16),
    B { one: u8, two: u16 },
}

assert_eq!(mem::offset_of!(Enum, A.0), 1);
assert_eq!(mem::offset_of!(Enum, B.two), 2);

assert_eq!(mem::offset_of!(Option<&u8>, Some.0), 0);
```

[`offset_of!`]: ../../std/mem/macro.offset_of.html
