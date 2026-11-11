//@ compile-flags: -Zunleash-the-miri-inside-of-you
//@ aux-build:static_cross_crate.rs
//@ normalize-stderr: "(the raw bytes of the constant) \(size: [0-9]*, align: [0-9]*\)" -> "$1 (size: $$SIZE, align: $$ALIGN)"
//@ normalize-stderr: "([0-9a-f][0-9a-f] |╾─*ALLOC[0-9]+(\+[a-z0-9]+)?(<imm>)?─*╼ )+ *│.*" -> "HEX_DUMP"

#![feature(half_open_range_patterns_in_slices)]
#![allow(static_mut_refs)]

extern crate static_cross_crate;

// Sneaky: reference to a mutable static.
// Allowing this would be a disaster for pattern matching, we could violate exhaustiveness checking!
const SLICE_MUT: &[u8; 1] = {
    unsafe { &static_cross_crate::ZERO }
};

const U8_MUT: &u8 = {
    unsafe { &static_cross_crate::ZERO[0] }
};

// Also test indirection that reads from other static.
const U8_MUT2: &u8 = {
    unsafe { &(*static_cross_crate::ZERO_REF)[0] }
};
const U8_MUT3: &u8 = {
    unsafe {
        match static_cross_crate::OPT_ZERO {
            //~^ ERROR constant accesses mutable global memory
            Some(ref u) => u,
            None => panic!(),
        }
    }
};

pub fn test(x: &[u8; 1]) -> bool {
    match x {
        SLICE_MUT => true, //~ ERROR cannot be used as pattern
        &[1..] => false,
    }
}

pub fn test2(x: &u8) -> bool {
    match x {
        U8_MUT => true, //~ ERROR cannot be used as pattern
        &(1..) => false,
    }
}

// We need to use these *in a pattern* to trigger the failure... likely because
// the errors above otherwise stop compilation too early?
pub fn test3(x: &u8) -> bool {
    match x {
        U8_MUT2 => true, //~ ERROR cannot be used as pattern
        &(1..) => false,
    }
}
pub fn test4(x: &u8) -> bool {
    match x {
        U8_MUT3 => true, // ok, `const` error already emitted
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
