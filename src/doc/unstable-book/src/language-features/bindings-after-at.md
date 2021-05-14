# `bindings_after_at`

The tracking issue for this feature is [#65490]

[#65490]: https://github.com/rust-lang/rust/issues/65490

------------------------


The `bindings_after_at` feature gate allows patterns of form `binding @ pat` to have bindings in `pat`.

```rust
#![feature(bindings_after_at)]

struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let point@ Point{x: px, y: py} = Point {x: 12, y: 34};
}
```
