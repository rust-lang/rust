//@ aux-build:cross_crate_debuginfo_type_uniquing.rs
extern crate cross_crate_debuginfo_type_uniquing;

//@ no-prefer-dynamic
//@ compile-flags:-g -C lto
//@ ignore-backends: gcc

pub struct C;
pub fn p() -> C {
    C
}

fn main() { }
