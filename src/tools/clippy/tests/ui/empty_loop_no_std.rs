//@compile-flags: -Clink-arg=-nostartfiles
//@ignore-target: apple

#![warn(clippy::empty_loop)]
#![crate_type = "lib"]
#![no_std]

pub fn main(argc: isize, argv: *const *const u8) -> isize {
    // This should trigger the lint
    loop {}
    //~^ empty_loop
}
