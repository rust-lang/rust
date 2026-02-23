// Just a test of the error message (it's different for non-local consts)
//@ aux-build:non_local_const.rs
#![feature(min_generic_const_args)]
#![allow(incomplete_features)]
extern crate non_local_const;
fn main() {
    let x = [(); non_local_const::N];
    //~^ ERROR: use of `const` in the type system not defined as `type const`
}
