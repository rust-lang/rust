# `transparent_enums`

The tracking issue for this feature is [#60405]

[60405]: https://github.com/rust-lang/rust/issues/60405

----

The `transparent_enums` feature allows you mark `enum`s as
`#[repr(transparent)]`. An `enum` may be `#[repr(transparent)]` if it has
exactly one variant, and that variant matches the same conditions which `struct`
requires for transparency. Some concrete illustrations follow.

```rust
#![feature(transparent_enums)]

// This enum has the same representation as `f32`.
#[repr(transparent)]
enum SingleFieldEnum {
    Variant(f32)
}

// This enum has the same representation as `usize`.
#[repr(transparent)]
enum MultiFieldEnum {
    Variant { field: usize, nothing: () },
}
```

For consistency with transparent `struct`s, `enum`s must have exactly one
non-zero-sized field. If all fields are zero-sized, the `enum` must not be
`#[repr(transparent)]`:

```rust
#![feature(transparent_enums)]

// This (non-transparent) enum is already valid in stable Rust:
pub enum GoodEnum {
    Nothing,
}

// Error: transparent enum needs exactly one non-zero-sized field, but has 0
// #[repr(transparent)]
// pub enum BadEnum {
//     Nothing(()),
// }

// Error: transparent enum needs exactly one non-zero-sized field, but has 0
// #[repr(transparent)]
// pub enum BadEmptyEnum {
//     Nothing,
// }
```

The one exception is if the `enum` is generic over `T` and has a field of type
`T`, it may be `#[repr(transparent)]` even if `T` is a zero-sized type:

```rust
#![feature(transparent_enums)]

// This enum has the same representation as `T`.
#[repr(transparent)]
pub enum GenericEnum<T> {
    Variant(T, ()),
}

// This is okay even though `()` is a zero-sized type.
pub const THIS_IS_OKAY: GenericEnum<()> = GenericEnum::Variant((), ());
```

Transparent `enum`s require exactly one variant:

```rust
// Error: transparent enum needs exactly one variant, but has 0
// #[repr(transparent)]
// pub enum TooFewVariants {
// }

// Error: transparent enum needs exactly one variant, but has 2
// #[repr(transparent)]
// pub enum TooManyVariants {
//     First(usize),
//     Second,
// }
```

Like transarent `struct`s, a transparent `enum` of type `E` has the same layout,
size, and ABI as its single non-ZST field. If it is generic over a type `T`, and
all its fields are ZSTs except for exactly one field of type `T`, then it has
the same layout and ABI as `T` (even if `T` is a ZST when monomorphized).

Like transparent `struct`s, transparent `enum`s are FFI-safe if and only if
their underlying representation type is also FFI-safe.
