// compile-flags: -Zunleash-the-miri-inside-of-you
// aux-build:static_cross_crate.rs
#![allow(const_err)]

#![feature(exclusive_range_pattern)]
#![feature(half_open_range_patterns)]

extern crate static_cross_crate;

// Sneaky: reference to a mutable static.
// Allowing this would be a disaster for pattern matching, we could violate exhaustiveness checking!
const SLICE_MUT: &[u8; 1] = { //~ ERROR undefined behavior to use this value
//~| NOTE encountered a reference pointing to a static variable
//~| NOTE
    unsafe { &static_cross_crate::ZERO }
    //~^ WARN skipping const checks
    //~| WARN skipping const checks
};

pub fn test(x: &[u8; 1]) -> bool {
    match x {
        SLICE_MUT => true,
        //~^ ERROR could not evaluate constant pattern
        //~| ERROR could not evaluate constant pattern
        &[1..] => false,
    }
}

fn main() {
    unsafe {
        static_cross_crate::ZERO[0] = 1;
    }
    // Now the pattern is not exhaustive any more!
    test(&[0]);
}
