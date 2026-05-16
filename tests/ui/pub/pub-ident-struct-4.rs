//@ run-rustfix

#![allow(unused_unconstructable_pub_structs)]
pub T(#[allow(dead_code)] String);
//~^ ERROR missing `struct` for struct definition

fn main() {}
