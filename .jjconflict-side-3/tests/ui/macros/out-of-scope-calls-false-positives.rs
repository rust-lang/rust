//@ check-pass
//@ needs-asm-support

macro_rules! mac { () => { "" } }
macro_rules! mac2 { () => { "auxiliary/issue-40469.rs" } }

std::arch::global_asm!(mac!()); // OK
include!(mac2!()); // OK

fn main() {}
