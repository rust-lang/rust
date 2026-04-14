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

/// Read all of `in_fd` into a Vec<u8>, then write the last `n` lines to stdout.
fn tail_fd(in_fd: u32, n: usize) {
    // Read entire input into memory (reasonable for OS utilities on small files)
    let mut data: Vec<u8> = Vec::new();
    let mut buf = alloc::vec![0u8; 4096];
    loop {
        match vfs_read(in_fd, &mut buf) {
            Ok(0) => break,
            Ok(count) => data.extend_from_slice(&buf[..count]),
            Err(_) => break,
        }
    }

    if data.is_empty() {
        return;
    }

    // Count newline positions from the end
    // We want the start offset of the (n)-th-from-last line.
    let mut newlines_found = 0usize;
    let mut start_offset = 0usize;

    // If data ends with a newline, don't count that as a line separator for our purposes
    let search_end = if data.last() == Some(&b'\n') && data.len() > 1 {
        data.len() - 1
    } else {
        data.len()
    };

    let mut i = search_end;
    loop {
        if i == 0 {
            start_offset = 0;
            break;
        }
        i -= 1;
        if data[i] == b'\n' {
            newlines_found += 1;
            if newlines_found == n {
                start_offset = i + 1;
                break;
            }
        }
    }

    let _ = vfs_write(1, &data[start_offset..]);
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
                    print_str(2, "tail: invalid number of lines\n");
                    stem::syscall::exit(1);
                }
            }
        } else if args[i].starts_with("-n") {
            let rest = &args[i][2..];
            if let Ok(v) = rest.parse::<usize>() {
                n = v;
            } else {
                print_str(2, "tail: invalid number of lines\n");
                stem::syscall::exit(1);
            }
        } else {
            file_args.push(&args[i]);
        }
        i += 1;
    }

    if file_args.is_empty() {
        tail_fd(0, n);
    } else {
        for path in &file_args {
            if *path == "-" {
                tail_fd(0, n);
            } else {
                match vfs_open(path, vfs_flags::O_RDONLY) {
                    Ok(fd) => {
                        tail_fd(fd, n);
                        let _ = vfs_close(fd);
                    }
                    Err(_) => {
                        let msg = alloc::format!("tail: {}: No such file or directory\n", path);
                        print_str(2, &msg);
                    }
                }
            }
        }
    }

    stem::syscall::exit(0)
}
