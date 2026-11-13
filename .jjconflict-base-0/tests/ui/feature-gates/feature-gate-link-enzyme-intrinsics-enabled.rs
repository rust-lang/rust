//@ check-pass
// gate-test-link_enzyme_intrinsics
#![feature(link_enzyme_intrinsics)]

unsafe extern "C" {
    fn __enzyme_autodiff();
    fn __enzyme_fwddiff();
    fn __enzyme_augmentfwd();
    fn __enzyme_reverse();
    #[link_name = "__enzyme_autodiff"]
    fn autodiff();

    static __enzyme_dup: i32;
    #[link_name = "__enzyme_dup"]
    static ENZYME_DUP: i32;
}

fn main() {}
