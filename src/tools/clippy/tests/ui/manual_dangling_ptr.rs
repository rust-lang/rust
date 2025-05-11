#![warn(clippy::manual_dangling_ptr)]
use std::mem;

pub fn foo(_const: *const f32, _mut: *mut i32) {}

fn main() {
    let _: *const u8 = 1 as *const _;
    //~^ manual_dangling_ptr
    let _ = 2 as *const u32;
    //~^ manual_dangling_ptr
    let _ = 4 as *mut f32;
    //~^ manual_dangling_ptr

    let _ = mem::align_of::<u8>() as *const u8;
    //~^ manual_dangling_ptr
    let _ = mem::align_of::<u32>() as *const u32;
    //~^ manual_dangling_ptr
    let _ = mem::align_of::<usize>() as *const usize;
    //~^ manual_dangling_ptr

    foo(4 as *const _, 4 as *mut _);
    //~^ manual_dangling_ptr
    //~| manual_dangling_ptr
}

fn should_not_lint() {
    let _ = 0x10 as *mut i32;
    let _ = mem::align_of::<u32>() as *const u8;

    foo(0 as _, 0 as _);
}

#[clippy::msrv = "1.83"]
fn _msrv_1_83() {
    // `{core, std}::ptr::dangling` was stabilized in 1.84. Do not lint this
    foo(4 as *const _, 4 as *mut _);
}

#[clippy::msrv = "1.84"]
fn _msrv_1_84() {
    foo(4 as *const _, 4 as *mut _);
    //~^ manual_dangling_ptr
    //~| manual_dangling_ptr
}
