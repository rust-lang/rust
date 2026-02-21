// Regression test for #149156.
//@ build-fail
//@ compile-flags: -Zno-codegen
//@ only-x86_64

#![crate_type = "lib"]
#![allow(private_interfaces)]

struct Struct([u8; 0xffff_ffff_ffff_ffff]);

pub fn function(value: Struct) -> u8 { //~ ERROR are too big for the target architecture
    value.0[0]
}
