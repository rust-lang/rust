//! Smoke test: argv comes through correctly via SYS_ARGV_GET.
//!
//! Acceptance criteria:
//!   - argv[0] is the executable name
//!   - subsequent arguments are preserved in order
#![no_std]
#![no_main]
extern crate alloc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::default::Default;

#[stem::main]
fn main() -> ! {
    let mut buf = [0u8; 4096];
    let n = match stem::syscall::argv_get(&mut buf) {
        Ok(n) => n,
        Err(_) => {
            stem::println!("[show_args] FAIL: could not get argv");
            loop { stem::syscall::exit(1); }
        }
    };

    if n < 4 {
        stem::println!("[show_args] FAIL: buf too small for count");
        loop { stem::syscall::exit(1); }
    }

    let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    stem::println!("[show_args] argc = {}", count);

    let mut offset = 4;
    for i in 0..count {
        if offset + 4 > n {
            stem::println!("[show_args] FAIL: out of bounds at index {}", i);
            loop { stem::syscall::exit(1); }
        }
        let len = u32::from_le_bytes(buf[offset..offset+4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + len > n {
            stem::println!("[show_args] FAIL: out of bounds for data at index {}", i);
            loop { stem::syscall::exit(1); }
        }
        let arg = core::str::from_utf8(&buf[offset..offset+len]).unwrap_or("<invalid>");
        stem::println!("[show_args] argv[{}] = {:?}", i, arg);
        offset += len;
    }

    if count == 0 {
        stem::println!("[show_args] FAIL: expected at least argv[0]");
        loop { stem::syscall::exit(1); }
    }

    stem::println!("[show_args] PASS");
    loop { stem::syscall::exit(0); }
}
