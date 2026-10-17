//! Regression test for <https://github.com/rust-lang/rust/issues/52141>.
//! Test wildcard import resolves correctly while renamed.
//@ run-pass
//@ aux-build:glob-import-renamed-crate.rs
//@ compile-flags:--extern glob_import_renamed_crate
//@ edition:2018

use glob_import_renamed_crate as some_name;

mod foo {
    pub use crate::some_name::*;
}

fn main() {
    ::glob_import_renamed_crate::hello();
    some_name::hello();
    foo::hello();
}
