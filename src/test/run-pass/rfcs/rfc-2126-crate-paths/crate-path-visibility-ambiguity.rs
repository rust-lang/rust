// run-pass
#![feature(crate_in_paths)]
#![feature(crate_visibility_modifier)]
#![allow(dead_code)]
mod m {
    pub struct Z;
    pub struct S1(crate (::m::Z)); // OK
    pub struct S2((crate ::m::Z)); // OK
    pub struct S3(crate ::m::Z); // OK
    pub struct S4(crate crate::m::Z); // OK
}

fn main() {
    crate struct S; // OK (item in statement position)
}
