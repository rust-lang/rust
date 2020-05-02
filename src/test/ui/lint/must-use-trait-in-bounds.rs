//! Tests that `T: MustUse` bounds do emit the lint.
//!
//! This is rather useless, but at least shouldn't ICE. Maybe it should be forbidden to use
//! `MustUse` in bounds.

#![feature(must_use_trait)]
#![deny(unused_must_use)] //~ NOTE the lint level is defined here

use std::marker::MustUse;

trait Tr: MustUse {}

fn f<T: MustUse>(t: T) {
    t;
    //~^ ERROR unused `T` that must be used
}

fn main() {}
