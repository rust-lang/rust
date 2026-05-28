#![warn(clippy::borrow_as_ptr)]
#![no_std]
#![crate_type = "lib"]

#[clippy::msrv = "1.75"]
pub fn main(_argc: isize, _argv: *const *const u8) -> isize {
    let val = 1;
    let _p = &val as *const i32;
    //~^ borrow_as_ptr

    let mut val_mut = 1;
    let _p_mut = &mut val_mut as *mut i32;
    //~^ borrow_as_ptr
    0
}
