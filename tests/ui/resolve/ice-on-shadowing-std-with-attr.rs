//@ edition: 2024
// This test ensures that `extern crate` with attribute shadowing std does not cause ICE.
// Issue link: https://github.com/rust-lang/rust/issues/152895

#![crate_type = "lib"]

#[foobar] //~ ERROR cannot find attribute `foobar` in this scope
extern crate core as std; //~ ERROR macro-expanded `extern crate` items cannot shadow names passed with `--extern`
mod inner {
    use std::collections::hash_map::HashMap; //~ ERROR cannot find `collections` in `std`
    use std::vec::IntoIter; //~ ERROR unresolved import `std::vec`

    use crate::*;
}
