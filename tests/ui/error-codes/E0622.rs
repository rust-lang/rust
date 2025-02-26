#![feature(intrinsics)]
extern "rust-intrinsic" {
    pub static atomic_singlethreadfence_seqcst : unsafe extern "rust-intrinsic" fn();
    //~^ ERROR intrinsic must be a function [E0622]
}
fn main() { unsafe { atomic_singlethreadfence_seqcst(); } }
