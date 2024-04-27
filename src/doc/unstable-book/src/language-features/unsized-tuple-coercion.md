# `unsized_tuple_coercion`

The tracking issue for this feature is: [#42877]

[#42877]: https://github.com/rust-lang/rust/issues/42877

------------------------

This is a part of [RFC0401]. According to the RFC, there should be an implementation like this:

```rust,ignore (partial-example)
impl<..., T, U: ?Sized> Unsized<(..., U)> for (..., T) where T: Unsized<U> {}
```

This implementation is currently gated behind `#[feature(unsized_tuple_coercion)]` to avoid insta-stability. Therefore you can use it like this:

```rust
#![feature(unsized_tuple_coercion)]

fn main() {
    let x : ([i32; 3], [i32; 3]) = ([1, 2, 3], [4, 5, 6]);
    let y : &([i32; 3], [i32]) = &x;
    assert_eq!(y.1[0], 4);
}
```

[RFC0401]: https://github.com/rust-lang/rfcs/blob/master/text/0401-coercions.md
