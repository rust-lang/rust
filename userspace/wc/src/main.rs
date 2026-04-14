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

struct Counts {
    lines: usize,
    words: usize,
    bytes: usize,
}

/// Count lines, words, and bytes in the data read from `in_fd`.
fn count_fd(in_fd: u32) -> Counts {
    let mut lines = 0usize;
    let mut words = 0usize;
    let mut bytes = 0usize;
    let mut in_word = false;
    let mut buf = alloc::vec![0u8; 4096];

    loop {
        match vfs_read(in_fd, &mut buf) {
            Ok(0) => break,
            Ok(count) => {
                bytes += count;
                for &b in &buf[..count] {
                    if b == b'\n' {
                        lines += 1;
                    }
                    let is_space = b == b' ' || b == b'\t' || b == b'\n' || b == b'\r';
                    if is_space {
                        if in_word {
                            words += 1;
                            in_word = false;
                        }
                    } else {
                        in_word = true;
                    }
                }
            }
            Err(_) => break,
        }
    }
    // Finish a word that reached EOF without whitespace
    if in_word {
        words += 1;
    }
    Counts { lines, words, bytes }
}

fn print_counts(counts: &Counts, show_lines: bool, show_words: bool, show_bytes: bool, label: &str) {
    let mut out = String::new();
    if show_lines {
        out.push_str(&alloc::format!("{:>8}", counts.lines));
    }
    if show_words {
        out.push_str(&alloc::format!("{:>8}", counts.words));
    }
    if show_bytes {
        out.push_str(&alloc::format!("{:>8}", counts.bytes));
    }
    if !label.is_empty() {
        out.push(' ');
        out.push_str(label);
    }
    out.push('\n');
    print_str(1, &out);
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();
    let mut show_lines = false;
    let mut show_words = false;
    let mut show_bytes = false;
    let mut file_args: Vec<&str> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg.starts_with('-') && arg.len() > 1 && !arg.starts_with("--") {
            for ch in arg[1..].chars() {
                match ch {
                    'l' => show_lines = true,
                    'w' => show_words = true,
                    'c' => show_bytes = true,
                    _ => {
                        let msg = alloc::format!("wc: illegal option -- {}\n", ch);
                        print_str(2, &msg);
                        stem::syscall::exit(1);
                    }
                }
            }
        } else {
            file_args.push(arg);
        }
        i += 1;
    }

    // Default: show all three if no flags specified
    if !show_lines && !show_words && !show_bytes {
        show_lines = true;
        show_words = true;
        show_bytes = true;
    }

    if file_args.is_empty() {
        let counts = count_fd(0);
        print_counts(&counts, show_lines, show_words, show_bytes, "");
    } else {
        let mut total = Counts { lines: 0, words: 0, bytes: 0 };
        let multiple = file_args.len() > 1;

        for path in &file_args {
            if *path == "-" {
                let counts = count_fd(0);
                if multiple {
                    total.lines += counts.lines;
                    total.words += counts.words;
                    total.bytes += counts.bytes;
                }
                print_counts(&counts, show_lines, show_words, show_bytes, "-");
            } else {
                match vfs_open(path, vfs_flags::O_RDONLY) {
                    Ok(fd) => {
                        let counts = count_fd(fd);
                        let _ = vfs_close(fd);
                        if multiple {
                            total.lines += counts.lines;
                            total.words += counts.words;
                            total.bytes += counts.bytes;
                        }
                        print_counts(&counts, show_lines, show_words, show_bytes, path);
                    }
                    Err(_) => {
                        let msg = alloc::format!("wc: {}: No such file or directory\n", path);
                        print_str(2, &msg);
                    }
                }
            }
        }

        if multiple {
            print_counts(&total, show_lines, show_words, show_bytes, "total");
        }
    }

    stem::syscall::exit(0)
}
