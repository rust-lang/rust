# `const_indexing`

The tracking issue for this feature is: [#29947]

[#29947]: https://github.com/rust-lang/rust/issues/29947

------------------------

The `const_indexing` feature allows the constant evaluation of index operations
on constant arrays and repeat expressions.

## Examples

```rust
#![feature(const_indexing)]

const ARR: [usize; 5] = [1, 2, 3, 4, 5];
const ARR2: [usize; ARR[1]] = [42, 99];
```