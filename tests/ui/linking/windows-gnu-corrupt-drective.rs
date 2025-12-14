//@ only-windows-gnu
//@ build-fail
#![deny(linker_info)]
//~? ERROR Warning: .drectve
fn main() {}
