//@ run-pass
//@ aux-build:sepcomp_lib.rs

// Test linking against a library built with -C codegen-units > 1


extern crate sepcomp_lib;
use sepcomp_lib::a::one;
use sepcomp_lib::b::two;
use sepcomp_lib::c::three;

fn main() {
    assert_eq!(one(), 1);
    assert_eq!(two(), 2);
    assert_eq!(three(), 3);
}
