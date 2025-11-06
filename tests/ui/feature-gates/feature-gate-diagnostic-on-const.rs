//! This is an unusual feature gate test, as it doesn't test the feature
//! gate, but the fact that not adding the feature gate to
//! the aux crate will cause the diagnostic to not emit the
//! custom diagnostic message

//@ aux-build: diagnostic-on-const.rs

extern crate diagnostic_on_const;
use diagnostic_on_const::Foo;

const fn foo() {
    Foo == Foo;
    //~^ ERROR: cannot call non-const operator in constant functions
}

fn main() {}
