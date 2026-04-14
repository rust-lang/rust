#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use stem::syscall::{exit, read, write};

#[stem::main]
fn main() -> ! {
    let _ = write(1, b"stdio_demo: interactive stdio demo\n");
    let _ = write(2, b"stdio_demo: stderr is attached too\n");
    let _ = write(1, b"name> ");

    let mut buf = [0u8; 128];
    let len = read(0, &mut buf).unwrap_or(0);
    let line = trim_line_len(&buf[..len]);

    let _ = write(1, b"hello, ");
    let _ = write(1, &buf[..line]);
    let _ = write(1, b"\n");
    exit(0);
}

fn trim_line_len(bytes: &[u8]) -> usize {
    let mut len = bytes.len();
    while len > 0 && matches!(bytes[len - 1], b'\n' | b'\r') {
        len -= 1;
    }
    len
}
