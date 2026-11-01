//@ only-windows-gnu
//@ build-fail
//@ compile-flags: -C linker={{src-base}}/linking/auxiliary/fake-linker.ps1
#![deny(linker_info)]
//~? ERROR Warning: .drectve
fn main() {}
