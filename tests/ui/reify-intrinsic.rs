//@ check-fail

#![feature(core_intrinsics, intrinsics)]

fn a() {
    let _: unsafe extern "rust-intrinsic" fn(isize) -> usize = std::mem::transmute;
    //~^ ERROR cannot coerce
}

fn b() {
    let _ = std::mem::transmute as unsafe extern "rust-intrinsic" fn(isize) -> usize;
    //~^ ERROR casting
}

fn c() {
    let _: [unsafe extern "rust-intrinsic" fn(bool) -> bool; 2] = [
        std::intrinsics::likely, //~ ERROR cannot coerce
        std::intrinsics::unlikely,
    ];
}

fn main() {}
