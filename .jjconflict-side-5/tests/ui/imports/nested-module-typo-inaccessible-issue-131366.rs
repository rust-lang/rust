//! Regression test for https://github.com/rust-lang/rust/issues/131366.
//! Similar spelling does not make a private module or a non-module importable.

//@ edition: 2021
#![allow(unused_imports, dead_code)]

mod private {
    mod collections { pub struct Item; }
}
use private::collection::Item;
//~^ ERROR unresolved import `private::collection`

mod non_module {
    pub struct Collections;
}
use non_module::Collection::Item;
//~^ ERROR unresolved import `non_module::Collection`

fn main() {}
