#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use stem::abi::syscall::vfs_flags;
use stem::syscall::{argv_get, vfs_close, vfs_open, vfs_read, vfs_write};

fn get_args() -> Vec<String> {
    let mut len = 0;
    if let Ok(l) = argv_get(&mut []) {
        len = l;
    }
    if len == 0 {
        return Vec::new();
    }
    let mut buf = alloc::vec![0u8; len];
    if argv_get(&mut buf).is_err() {
        return Vec::new();
    }

    let mut args = Vec::new();
    if buf.len() >= 4 {
        let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let mut offset = 4;
        for _ in 0..count {
            if offset + 4 > buf.len() {
                break;
            }
            let str_len = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            if offset + str_len > buf.len() {
                break;
            }
            if let Ok(s) = core::str::from_utf8(&buf[offset..offset + str_len]) {
                args.push(String::from(s));
            }
            offset += str_len;
        }
    }
    args
}

fn print_error(msg: &str) {
    let out = alloc::format!("cat: {}\n", msg);
    let _ = vfs_write(1, out.as_bytes());
}

/// Stream bytes from in_fd to out_fd until EOF.
fn stream(in_fd: u32, out_fd: u32, buf: &mut [u8]) -> Result<(), ()> {
    loop {
        match vfs_read(in_fd, buf) {
            Ok(0) => break,
            Ok(n) => {
                let mut written = 0;
                while written < n {
                    match vfs_write(out_fd, &buf[written..n]) {
                        Ok(0) => {
                            // If we can't write any bytes and it's not an error,
                            // might be a full pipe or similar. Try again?
                            // For simplicity, treat as error if it persists.
                            return Err(());
                        }
                        Ok(nw) => {
                            written += nw;
                        }
                        Err(_) => return Err(()),
                    }
                }
            }
            Err(_) => return Err(()),
        }
    }
    Ok(())
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();
    let mut buf = alloc::vec![0u8; 32768]; // 32KB buffer

    if args.len() < 2 {
        // No files specified, read from stdin (fd 0)
        if stream(0, 1, &mut buf).is_err() {
            print_error("error reading from stdin");
        }
    } else {
        for path in args.iter().skip(1) {
            if path == "-" {
                // Special case: read from stdin
                if stream(0, 1, &mut buf).is_err() {
                    print_error("error reading from stdin");
                }
                continue;
            }

            match vfs_open(path, vfs_flags::O_RDONLY) {
                Ok(fd) => {
                    if stream(fd, 1, &mut buf).is_err() {
                        print_error(&alloc::format!("error reading {}", path));
                    }
                    let _ = vfs_close(fd);
                }
                Err(_) => {
                    print_error(&alloc::format!("failed to open {}", path));
                }
            }
        }
    }

    stem::syscall::exit(0)
}
