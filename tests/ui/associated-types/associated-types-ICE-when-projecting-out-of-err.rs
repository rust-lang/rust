//@ add-core-stubs
// Test that we do not ICE when the self type is `ty::err`, but rather
// just propagate the error.

#![crate_type = "lib"]
#![feature(lang_items)]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

fn ice<A>(a: A) {
    let r = loop {};
    r = r + a;
    //~^ ERROR the trait bound `(): Add<A>` is not satisfied
}
