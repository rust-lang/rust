//! Regression test for <https://github.com/rust-lang/rust/issues/21202>.
//! Tests cross-crate ufcs doesn't bypass privacy checks.
//!
//@ aux-build:ufcs-cross-crate.rs

extern crate ufcs_cross_crate as crate1;

use crate1::A;

mod B {
    use crate1::A::Foo;
    fn bar(f: Foo) {
        Foo::foo(&f);
        //~^ ERROR: method `foo` is private
    }
}

fn main() { }
