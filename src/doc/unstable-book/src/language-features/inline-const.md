# `inline_const`

The tracking issue for this feature is: [#76001]

See also [`inline_const_pat`](inline-const-pat.md)

------

This feature allows you to use inline constant expressions. For example, you can
turn this code:

```rust
# fn add_one(x: i32) -> i32 { x + 1 }
const MY_COMPUTATION: i32 = 1 + 2 * 3 / 4;

fn main() {
    let x = add_one(MY_COMPUTATION);
}
```

into this code:

```rust
#![feature(inline_const)]

# fn add_one(x: i32) -> i32 { x + 1 }
fn main() {
    let x = add_one(const { 1 + 2 * 3 / 4 });
}
```

[#76001]: https://github.com/rust-lang/rust/issues/76001
