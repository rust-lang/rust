#![feature(intrinsics)]
#![feature(rustc_attrs)]

extern "rust-intrinsic" {
    fn size_of<T>() -> usize; //~ ERROR intrinsic safety mismatch

    #[rustc_safe_intrinsic]
    fn assume(b: bool); //~ ERROR intrinsic safety mismatch
}

fn main() {}
