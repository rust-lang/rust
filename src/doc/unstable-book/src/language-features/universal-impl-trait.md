# `universal_impl_trait`

The tracking issue for this feature is: [#34511].

[#34511]: https://github.com/rust-lang/rust/issues/34511

--------------------

The `universal_impl_trait` feature extends the [`conservative_impl_trait`]
feature allowing the `impl Trait` syntax in arguments (universal
quantification).

[`conservative_impl_trait`]: ./language-features/conservative-impl-trait.html

## Examples

```rust
#![feature(universal_impl_trait)]
use std::ops::Not;

fn any_zero(values: impl IntoIterator<Item = i32>) -> bool {
    for val in values { if val == 0 { return true; } }
    false
}

fn main() {
    let test1 = -5..;
    let test2 = vec![1, 8, 42, -87, 60];
    assert!(any_zero(test1));
    assert!(bool::not(any_zero(test2)));
}
```
