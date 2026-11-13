#![feature(intrinsics)]
#![feature(rustc_attrs)]

#[rustc_intrinsic]
unsafe fn size_of<T>() -> usize;
//~^ ERROR intrinsic safety mismatch
//~| ERROR intrinsic has wrong type

#[rustc_intrinsic]
const fn assume(_b: bool) {}
//~^ ERROR intrinsic safety mismatch
//~| ERROR intrinsic has wrong type

#[rustc_intrinsic]
const fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize) {}
//~^ ERROR intrinsic safety mismatch
//~| ERROR intrinsic has wrong type

mod foo {
    #[rustc_intrinsic]
    unsafe fn const_deallocate(_ptr: *mut u8, _size: usize, _align: usize) {}
}

fn main() {}
