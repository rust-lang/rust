# `iterator_for_each`

The tracking issue for this feature is: [#TBD]

[#TBD]: https://github.com/rust-lang/rust/issues/TBD

------------------------

To call a closure on each element of an iterator, you can use `for_each`:

```rust
#![feature(iterator_for_each)]

fn main() {
    (0..10).for_each(|i| println!("{}", i));
}
```
