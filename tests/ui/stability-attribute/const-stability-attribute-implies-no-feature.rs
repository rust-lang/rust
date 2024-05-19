//@ aux-build:const-stability-attribute-implies.rs
#![crate_type = "lib"]

// Tests that despite the `const_foobar` feature being implied by now-stable feature `const_foo`,
// if `const_foobar` isn't allowed in this crate then an error will be emitted.

extern crate const_stability_attribute_implies;
use const_stability_attribute_implies::{foo, foobar};

pub const fn bar() -> u32 {
    foo(); // no error - stable
    foobar(); //~ ERROR `foobar` is not yet stable as a const fn
    0
}

pub const VAR: u32 = bar();
