# `match_beginning_vert`

The tracking issue for this feature is [#44101].

With this feature enabled, you are allowed to add a '|' to the beginning of a
match arm:

```rust
#![feature(match_beginning_vert)]

enum Foo { A, B }

fn main() {
    let x = Foo::A;
    match x {
        | A | B => {},
    }
}
```

[#44101]: https://github.com/rust-lang/rust/issues/44101