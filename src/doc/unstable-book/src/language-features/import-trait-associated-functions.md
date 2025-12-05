# import_trait_associated_functions

The tracking issue for this feature is: [#134691]

[#134691]: https://github.com/rust-lang/rust/issues/134691

------------------------

This feature allows importing associated functions and constants from traits and then using them like regular items.

```rust
#![feature(import_trait_associated_functions)]

use std::ops::Add::add;

fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6];
    let sum = numbers.into_iter().reduce(add); // instead of `.reduce(Add:add)`

    assert_eq!(sum, Some(21));
}
```
