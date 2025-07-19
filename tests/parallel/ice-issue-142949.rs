// Test for #142949, which causes an ice bug, "could not send CguMessage to main thread".
//
//@ compile-flags: -Z threads=6 -Z validate-mir -Z dump-mir-dir=.

#![crate_type = "lib"]

struct Struct<const N: i128>(pub [u8; 0xffff_ffff_ffff_ffff]);

pub fn function(value: Struct<3>) -> u8 {
    value.0[0]
}
