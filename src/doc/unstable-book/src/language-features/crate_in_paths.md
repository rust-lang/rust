# `crate_in_paths`

The tracking issue for this feature is: [#44660]

[#44660]: https://github.com/rust-lang/rust/issues/44660

------------------------

The `crate_in_paths` feature allows to explicitly refer to the crate root in absolute paths
using keyword `crate`.

`crate` can be used *only* in absolute paths, i.e. either in `::crate::a::b::c` form or in `use`
items where the starting `::` is added implicitly.  
Paths like `crate::a::b::c` are not accepted currently.

This feature is required in `feature(extern_absolute_paths)` mode to refer to any absolute path
in the local crate (absolute paths refer to extern crates by default in that mode), but can be
used without `feature(extern_absolute_paths)` as well.

```rust
#![feature(crate_in_paths)]

// Imports, `::` is added implicitly
use crate::m::f;
use crate as root;

mod m {
    pub fn f() -> u8 { 1 }
    pub fn g() -> u8 { 2 }
    pub fn h() -> u8 { 3 }

    // OK, visibilities implicitly add starting `::` as well, like imports
    pub(in crate::m) struct S;
}

mod n
{
    use crate::m::f;
    use crate as root;
    pub fn check() {
        assert_eq!(f(), 1);
        // `::` is required in non-import paths
        assert_eq!(::crate::m::g(), 2);
        assert_eq!(root::m::h(), 3);
    }
}

fn main() {
    assert_eq!(f(), 1);
    assert_eq!(::crate::m::g(), 2);
    assert_eq!(root::m::h(), 3);
    n::check();
}
```
