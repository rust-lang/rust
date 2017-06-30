# `iterator_for_each`

The tracking issue for this feature is: [#42986]

[#42986]: https://github.com/rust-lang/rust/issues/42986

------------------------

To call a closure on each element of an iterator, you can use `for_each`:

```rust
#![feature(iterator_for_each)]

fn main() {
    (0..10).for_each(|i| println!("{}", i));
}
```
