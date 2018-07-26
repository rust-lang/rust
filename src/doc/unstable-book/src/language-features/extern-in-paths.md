# `extern_in_paths`

The tracking issue for this feature is: [#44660]

[#44660]: https://github.com/rust-lang/rust/issues/44660

------------------------

The `extern_in_paths` feature allows to refer to names from other crates "inline", without
introducing `extern crate` items, using keyword `extern`.

For example, `extern::my_crat::a::b` will resolve to path `a::b` in crate `my_crate`.

`feature(extern_absolute_paths)` mode provides the same effect by resolving absolute paths like
`::my_crate::a::b` to paths from extern crates by default.

```rust,ignore
#![feature(extern_in_paths)]

// Suppose we have a dependency crate `xcrate` available through `Cargo.toml`, or `--extern`
// options, or standard Rust distribution, or some other means.

use extern::xcrate::Z;

fn f() {
    use extern::xcrate;
    use extern::xcrate as ycrate;
    let s = xcrate::S;
    assert_eq!(format!("{:?}", s), "S");
    let z = ycrate::Z;
    assert_eq!(format!("{:?}", z), "Z");
}

fn main() {
    let s = extern::xcrate::S;
    assert_eq!(format!("{:?}", s), "S");
    let z = Z;
    assert_eq!(format!("{:?}", z), "Z");
}
```
