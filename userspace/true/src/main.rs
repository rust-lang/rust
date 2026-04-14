#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

#[stem::main]
fn main(_arg: usize) -> ! {
    stem::syscall::exit(0)
}
