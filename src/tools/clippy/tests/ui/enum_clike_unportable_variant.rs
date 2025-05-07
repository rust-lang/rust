//@ignore-bitwidth: 32

#![warn(clippy::enum_clike_unportable_variant)]
#![allow(unused, non_upper_case_globals)]

#[repr(usize)]
enum NonPortable {
    X = 0x1_0000_0000,
    //~^ enum_clike_unportable_variant
    Y = 0,
    Z = 0x7FFF_FFFF,
    A = 0xFFFF_FFFF,
}

enum NonPortableNoHint {
    X = 0x1_0000_0000,
    //~^ enum_clike_unportable_variant
    Y = 0,
    Z = 0x7FFF_FFFF,
    A = 0xFFFF_FFFF,
    //~^ enum_clike_unportable_variant
}

#[repr(isize)]
enum NonPortableSigned {
    X = -1,
    Y = 0x7FFF_FFFF,
    Z = 0xFFFF_FFFF,
    //~^ enum_clike_unportable_variant
    A = 0x1_0000_0000,
    //~^ enum_clike_unportable_variant
    B = i32::MIN as isize,
    C = (i32::MIN as isize) - 1,
    //~^ enum_clike_unportable_variant
}

enum NonPortableSignedNoHint {
    X = -1,
    Y = 0x7FFF_FFFF,
    Z = 0xFFFF_FFFF,
    //~^ enum_clike_unportable_variant
    A = 0x1_0000_0000,
    //~^ enum_clike_unportable_variant
}

#[repr(usize)]
enum NonPortable2 {
    X = <usize as Trait>::Number,
    //~^ enum_clike_unportable_variant
    Y = 0,
}

trait Trait {
    const Number: usize = 0x1_0000_0000;
}

impl Trait for usize {}

fn main() {}
