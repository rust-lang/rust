#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use stem::syscall::vfs_write;

#[stem::main]
fn main(_arg: usize) -> ! {
    // ANSI clear screen: \x1B[2J
    // ANSI move cursor to top-left: \x1B[H
    let _ = vfs_write(1, b"\x1B[2J\x1B[H");

    stem::syscall::exit(0)
}
