#![feature(intrinsics)]

extern "C" {

    #[rustc_intrinsic]
    pub static atomic_singlethreadfence_seqcst: unsafe extern "C" fn();
    //~^ ERROR intrinsic must be a function [E0622]
}

fn main() {
    unsafe {
        atomic_singlethreadfence_seqcst();
    }
}
