//@ run-rustfix

#![allow(unconstructable_pub_struct)]

pub T(#[allow(dead_code)] String);
//~^ ERROR missing `struct` for struct definition

fn main() {}
