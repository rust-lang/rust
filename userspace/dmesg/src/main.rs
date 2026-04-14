#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use stem::abi::syscall::vfs_flags;
use stem::syscall::{vfs_close, vfs_open, vfs_read, vfs_write};

#[stem::main]
fn main(_arg: usize) -> ! {
    let mut buf = alloc::vec![0u8; 32768]; // 32KB buffer

    match vfs_open("/dev/kmsg", vfs_flags::O_RDONLY) {
        Ok(fd) => {
            loop {
                match vfs_read(fd, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let mut written = 0;
                        while written < n {
                            match vfs_write(1, &buf[written..n]) {
                                Ok(0) => break,
                                Ok(nw) => written += nw,
                                Err(_) => break,
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            let _ = vfs_close(fd);
        }
        Err(_) => {
            let _ = vfs_write(1, b"dmesg: failed to open /dev/kmsg\n");
        }
    }

    stem::syscall::exit(0)
}
