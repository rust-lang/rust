//@ edition: 2024
// This test ensures that a macro-expanded `extern crate` shadowing std does not cause ICE.
// Issue link: https://github.com/rust-lang/rust/issues/152895

#![crate_type = "lib"]

mod inner {
    use std::collections::hash_map::HashMap; //~ ERROR cannot find `collections` in `std`
    use std::vec::IntoIter; //~ ERROR unresolved import `std::vec`

    use crate::*;
}

macro_rules! define_other_core {
    () => {
        extern crate core as std; //~ ERROR macro-expanded `extern crate` items cannot shadow names passed with `--extern`
    };
}
define_other_core! {}
