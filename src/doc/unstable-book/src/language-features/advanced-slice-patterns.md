# `advanced_slice_patterns`

The tracking issue for this feature is: [#23121]

[#23121]: https://github.com/rust-lang/rust/issues/23121

See also [`slice_patterns`](slice-patterns.html).

------------------------


The `advanced_slice_patterns` gate lets you use `..` to indicate any number of
elements inside a pattern matching a slice. This wildcard can only be used once
for a given array. If there's an identifier before the `..`, the result of the
slice will be bound to that name. For example:

```rust
#![feature(advanced_slice_patterns, slice_patterns)]

fn is_symmetric(list: &[u32]) -> bool {
    match list {
        &[] | &[_] => true,
        &[x, ref inside.., y] if x == y => is_symmetric(inside),
        _ => false
    }
}

fn main() {
    let sym = &[0, 1, 4, 2, 4, 1, 0];
    assert!(is_symmetric(sym));

    let not_sym = &[0, 1, 7, 2, 4, 1, 0];
    assert!(!is_symmetric(not_sym));
}
```
