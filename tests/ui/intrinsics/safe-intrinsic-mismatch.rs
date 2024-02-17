#![feature(intrinsics)]
#![feature(rustc_attrs)]
#![feature(effects)]

extern "rust-intrinsic" {
    fn size_of<T>() -> usize; //~ ERROR intrinsic safety mismatch
    //~^ ERROR intrinsic safety mismatch

    #[rustc_safe_intrinsic]
    fn assume(b: bool); //~ ERROR intrinsic safety mismatch
    //~^ ERROR intrinsic safety mismatch
}

#[rustc_intrinsic]
const fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize) {}
//~^ ERROR intrinsic safety mismatch
//~| ERROR intrinsic has wrong type

mod foo {
    #[rustc_intrinsic]
    unsafe fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize) {}
    //~^ ERROR wrong number of const parameters
}

fn main() {}
