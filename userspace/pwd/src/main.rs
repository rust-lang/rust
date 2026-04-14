#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use stem::syscall::{vfs_getcwd, vfs_write};

#[stem::main]
fn main(_arg: usize) -> ! {
    let mut buf = alloc::vec![0u8; 4096];
    match vfs_getcwd(&mut buf) {
        Ok(n) => {
            let _ = vfs_write(1, &buf[..n]);
            let _ = vfs_write(1, b"\n");
        }
        Err(_) => {
            let _ = vfs_write(2, b"pwd: error retrieving current directory\n");
        }
    }
    stem::syscall::exit(0)
}
