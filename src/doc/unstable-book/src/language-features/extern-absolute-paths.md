# `extern_absolute_paths`

The tracking issue for this feature is: [#44660]

[#44660]: https://github.com/rust-lang/rust/issues/44660

------------------------

The `extern_absolute_paths` feature enables mode allowing to refer to names from other crates
"inline", without introducing `extern crate` items, using absolute paths like `::my_crate::a::b`.

`::my_crate::a::b` will resolve to path `a::b` in crate `my_crate`.

`feature(crate_in_paths)` can be used in `feature(extern_absolute_paths)` mode for referring
to absolute paths in the local crate (`::crate::a::b`).

`feature(extern_in_paths)` provides the same effect by using keyword `extern` to refer to
paths from other crates (`extern::my_crate::a::b`).

```rust,ignore
#![feature(extern_absolute_paths)]

// Suppose we have a dependency crate `xcrate` available through `Cargo.toml`, or `--extern`
// options, or standard Rust distribution, or some other means.

use xcrate::Z;

fn f() {
    use xcrate;
    use xcrate as ycrate;
    let s = xcrate::S;
    assert_eq!(format!("{:?}", s), "S");
    let z = ycrate::Z;
    assert_eq!(format!("{:?}", z), "Z");
}

fn main() {
    let s = ::xcrate::S;
    assert_eq!(format!("{:?}", s), "S");
    let z = Z;
    assert_eq!(format!("{:?}", z), "Z");
}
```
