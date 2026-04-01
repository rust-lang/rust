#![crate_type = "lib"]
#![no_std]
#![deny(clippy::zero_ptr)]

pub fn main(_argc: isize, _argv: *const *const u8) -> isize {
    let _ = 0 as *const usize;
    //~^ zero_ptr
    let _ = 0 as *mut f64;
    //~^ zero_ptr
    let _: *const u8 = 0 as *const _;
    //~^ zero_ptr
    0
}
