// run-pass
// ignore-uefi allocation and other std functionality is intialized in `sys::init`. This test
// causes CPU Exception.

#![feature(start)]

#[start]
pub fn main(_: isize, _: *const *const u8) -> isize {
    println!("hello");
    0
}
