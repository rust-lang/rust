#![warn(clippy::useless_attribute)]

#[allow(dead_code)]
#[cfg_attr(feature = "cargo-clippy", allow(dead_code))]
#[rustfmt::skip]
#[cfg_attr(feature = "cargo-clippy",
           allow(dead_code))]
#[allow(unused_imports)]
#[allow(unused_extern_crates)]
#[macro_use]
extern crate clippy_lints;

// don't lint on unused_import for `use` items
#[allow(unused_imports)]
use std::collections;

// don't lint on deprecated for `use` items
mod foo {
    #[deprecated]
    pub struct Bar;
}
#[allow(deprecated)]
pub use foo::Bar;

fn main() {}
