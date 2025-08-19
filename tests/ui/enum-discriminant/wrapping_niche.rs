//! Test that we produce the same niche range no
//! matter of signendess if the discriminants are the same.

#![feature(rustc_attrs)]

#[repr(u16)]
#[rustc_layout(debug)]
enum UnsignedAroundZero {
    //~^ ERROR: layout_of
    A = 65535,
    B = 0,
    C = 1,
}

#[repr(i16)]
#[rustc_layout(debug)]
enum SignedAroundZero {
    //~^ ERROR: layout_of
    A = -1,
    B = 0,
    C = 1,
}

fn main() {}
