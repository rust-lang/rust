# `slice_patterns`

The tracking issue for this feature is: [#62254]

[#62254]: https://github.com/rust-lang/rust/issues/62254

------------------------

The `slice_patterns` feature gate lets you use `..` to indicate any number of
elements inside a pattern matching a slice. This wildcard can only be used once
for a given array. If there's an pattern before the `..`, the subslice will be
matched against that pattern. For example:

```rust
#![feature(slice_patterns)]

fn is_symmetric(list: &[u32]) -> bool {
    match list {
        &[] | &[_] => true,
        &[x, ref inside.., y] if x == y => is_symmetric(inside),
        &[..] => false,
    }
}

fn main() {
    let sym = &[0, 1, 4, 2, 4, 1, 0];
    assert!(is_symmetric(sym));

    let not_sym = &[0, 1, 7, 2, 4, 1, 0];
    assert!(!is_symmetric(not_sym));
}
```
