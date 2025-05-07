//@ check-fail

#![feature(core_intrinsics, intrinsics)]

fn a() {
    let _: unsafe fn(isize) -> usize = std::mem::transmute;
    //~^ ERROR cannot coerce
}

fn b() {
    let _ = std::mem::transmute as unsafe fn(isize) -> usize;
    //~^ ERROR casting
}

fn c() {
    let _: [unsafe fn(f32) -> f32; 2] = [
        std::intrinsics::floorf32, //~ ERROR cannot coerce
        std::intrinsics::log2f32,
    ];
}

fn main() {}
