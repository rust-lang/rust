// compile-flags: -Zunleash-the-miri-inside-of-you
// aux-build:static_cross_crate.rs
#![allow(const_err)]

// `const_if_match` is a HIR check and thus needed even when unleashed.
#![feature(exclusive_range_pattern, half_open_range_patterns, const_if_match)]

extern crate static_cross_crate;

// Sneaky: reference to a mutable static.
// Allowing this would be a disaster for pattern matching, we could violate exhaustiveness checking!
const SLICE_MUT: &[u8; 1] = { //~ ERROR undefined behavior to use this value
//~| NOTE encountered a reference pointing to a static variable
//~| NOTE
    unsafe { &static_cross_crate::ZERO }
};

const U8_MUT: &u8 = { //~ ERROR undefined behavior to use this value
//~| NOTE encountered a reference pointing to a static variable
//~| NOTE
    unsafe { &static_cross_crate::ZERO[0] }
};

// Also test indirection that reads from other static. This causes a const_err.
#[warn(const_err)] //~ NOTE
const U8_MUT2: &u8 = { //~ NOTE
    unsafe { &(*static_cross_crate::ZERO_REF)[0] }
    //~^ WARN [const_err]
    //~| NOTE constant accesses static
};
#[warn(const_err)] //~ NOTE
const U8_MUT3: &u8 = { //~ NOTE
    unsafe { match static_cross_crate::OPT_ZERO { Some(ref u) => u, None => panic!() } }
    //~^ WARN [const_err]
    //~| NOTE constant accesses static
};

pub fn test(x: &[u8; 1]) -> bool {
    match x {
        SLICE_MUT => true,
        //~^ ERROR could not evaluate constant pattern
        //~| ERROR could not evaluate constant pattern
        &[1..] => false,
    }
}

pub fn test2(x: &u8) -> bool {
    match x {
        U8_MUT => true,
        //~^ ERROR could not evaluate constant pattern
        //~| ERROR could not evaluate constant pattern
        &(1..) => false,
    }
}

// We need to use these *in a pattern* to trigger the failure... likely because
// the errors above otherwise stop compilation too early?
pub fn test3(x: &u8) -> bool {
    match x {
        U8_MUT2 => true,
        //~^ ERROR could not evaluate constant pattern
        //~| ERROR could not evaluate constant pattern
        &(1..) => false,
    }
}
pub fn test4(x: &u8) -> bool {
    match x {
        U8_MUT3 => true,
        //~^ ERROR could not evaluate constant pattern
        //~| ERROR could not evaluate constant pattern
        &(1..) => false,
    }
}

fn main() {
    unsafe {
        static_cross_crate::ZERO[0] = 1;
    }
    // Now the pattern is not exhaustive any more!
    test(&[0]);
    test2(&0);
}
