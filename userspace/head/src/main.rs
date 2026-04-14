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
            let str_len =
                u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
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

fn print_str(fd: u32, s: &str) {
    let _ = vfs_write(fd, s.as_bytes());
}

/// Print the first `n` lines from `in_fd` to stdout.
fn head_fd(in_fd: u32, n: usize) {
    let mut buf = alloc::vec![0u8; 4096];
    let mut lines_remaining = n;
    let mut leftover: Vec<u8> = Vec::new();

    'outer: loop {
        match vfs_read(in_fd, &mut buf) {
            Ok(0) => break,
            Ok(count) => {
                let chunk = &buf[..count];
                let combined_len = leftover.len() + chunk.len();
                let mut combined: Vec<u8> = Vec::with_capacity(combined_len);
                combined.extend_from_slice(&leftover);
                combined.extend_from_slice(chunk);
                leftover.clear();

                let mut start = 0;
                for i in 0..combined.len() {
                    if combined[i] == b'\n' {
                        let _ = vfs_write(1, &combined[start..=i]);
                        start = i + 1;
                        if lines_remaining <= 1 {
                            lines_remaining = 0;
                            break 'outer;
                        }
                        lines_remaining -= 1;
                    }
                }
                // Keep partial line for next iteration
                leftover.extend_from_slice(&combined[start..]);
            }
            Err(_) => break,
        }
    }

    // Flush any trailing partial line (no newline at end of file)
    if lines_remaining > 0 && !leftover.is_empty() {
        let _ = vfs_write(1, &leftover);
    }
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();
    let mut n: usize = 10;
    let mut file_args: Vec<&str> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        if args[i] == "-n" {
            i += 1;
            if i < args.len() {
                if let Ok(v) = args[i].parse::<usize>() {
                    n = v;
                } else {
                    print_str(2, "head: invalid number of lines\n");
                    stem::syscall::exit(1);
                }
            }
        } else if args[i].starts_with("-n") {
            let rest = &args[i][2..];
            if let Ok(v) = rest.parse::<usize>() {
                n = v;
            } else {
                print_str(2, "head: invalid number of lines\n");
                stem::syscall::exit(1);
            }
        } else {
            file_args.push(&args[i]);
        }
        i += 1;
    }

    if file_args.is_empty() {
        head_fd(0, n);
    } else {
        for path in &file_args {
            if *path == "-" {
                head_fd(0, n);
            } else {
                match vfs_open(path, vfs_flags::O_RDONLY) {
                    Ok(fd) => {
                        head_fd(fd, n);
                        let _ = vfs_close(fd);
                    }
                    Err(_) => {
                        let msg = alloc::format!("head: {}: No such file or directory\n", path);
                        print_str(2, &msg);
                    }
                }
            }
        }
    }

    stem::syscall::exit(0)
}
