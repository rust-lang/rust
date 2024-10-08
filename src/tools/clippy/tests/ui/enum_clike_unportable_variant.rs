//@ignore-bitwidth: 32

#![warn(clippy::enum_clike_unportable_variant)]
#![allow(unused, non_upper_case_globals)]

#[repr(usize)]
enum NonPortable {
    X = 0x1_0000_0000,
    //~^ ERROR: C-like enum variant discriminant is not portable to 32-bit targets
    //~| NOTE: `-D clippy::enum-clike-unportable-variant` implied by `-D warnings`
    Y = 0,
    Z = 0x7FFF_FFFF,
    A = 0xFFFF_FFFF,
}

enum NonPortableNoHint {
    X = 0x1_0000_0000,
    //~^ ERROR: C-like enum variant discriminant is not portable to 32-bit targets
    Y = 0,
    Z = 0x7FFF_FFFF,
    A = 0xFFFF_FFFF,
    //~^ ERROR: C-like enum variant discriminant is not portable to 32-bit targets
}

#[repr(isize)]
enum NonPortableSigned {
    X = -1,
    Y = 0x7FFF_FFFF,
    Z = 0xFFFF_FFFF,
    //~^ ERROR: C-like enum variant discriminant is not portable to 32-bit targets
    A = 0x1_0000_0000,
    //~^ ERROR: C-like enum variant discriminant is not portable to 32-bit targets
    B = i32::MIN as isize,
    C = (i32::MIN as isize) - 1,
    //~^ ERROR: C-like enum variant discriminant is not portable to 32-bit targets
}

enum NonPortableSignedNoHint {
    X = -1,
    Y = 0x7FFF_FFFF,
    Z = 0xFFFF_FFFF,
    //~^ ERROR: C-like enum variant discriminant is not portable to 32-bit targets
    A = 0x1_0000_0000,
    //~^ ERROR: C-like enum variant discriminant is not portable to 32-bit targets
}

#[repr(usize)]
enum NonPortable2 {
    X = <usize as Trait>::Number,
    //~^ ERROR: C-like enum variant discriminant is not portable to 32-bit targets
    Y = 0,
}

trait Trait {
    const Number: usize = 0x1_0000_0000;
}

impl Trait for usize {}

fn main() {}
