//@ check-pass
//@ aux-build:underscore-imports.rs

extern crate underscore_imports;

use underscore_imports::*;

fn main() {
    ().in_scope1();
    ().in_scope2();
}
