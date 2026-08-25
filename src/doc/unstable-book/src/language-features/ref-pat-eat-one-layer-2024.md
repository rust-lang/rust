# `ref_pat_eat_one_layer_2024`

The tracking issue for this feature is: [#123076]

[#123076]: https://github.com/rust-lang/rust/issues/123076

---

This feature is incomplete and not yet intended for general use.

This implements experimental, Edition-dependent match ergonomics under consideration for inclusion
in Rust, allowing `&` patterns in more places. For example:

```rust,edition2024
#![feature(ref_pat_eat_one_layer_2024)]
#![allow(incomplete_features)]
#
# // Tests type equality in a way that avoids coercing `&&T` or `&mut T` to `&T`.
# trait Eq<T> {}
# impl<T> Eq<T> for T {}
# fn has_type<T>(_: impl Eq<T>) {}

// `&` can match against a `ref` binding mode instead of a reference type:
let (x, &y) = &(0, 1);
has_type::<&u8>(x);
has_type::<u8>(y);

// `&` can match against `&mut` references:
let &z = &mut 2;
has_type::<u8>(z);
```

For specifics, see the corresponding typing rules for [Editions 2021 and earlier] and for
[Editions 2024 and later]. For more information on binding modes, see [The Rust Reference].

For alternative experimental match ergonomics, see the feature
[`ref_pat_eat_one_layer_2024_structural`](./ref-pat-eat-one-layer-2024-structural.md).

[Editions 2021 and earlier]: https://nadrieril.github.io/typing-rust-patterns/?compare=false&opts1=AQEBAQIBAQABAAAAAQEBAAEBAAABAAA%3D&mode=rules&do_cmp=false
[Editions 2024 and later]: https://nadrieril.github.io/typing-rust-patterns/?compare=false&opts1=AQEBAAABAQABAgIAAQEBAAEBAAABAAA%3D&mode=rules&do_cmp=false
[The Rust Reference]: https://doc.rust-lang.org/reference/patterns.html#binding-modes
