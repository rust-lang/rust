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

fn write_str(fd: u32, s: &str) {
    let _ = vfs_write(fd, s.as_bytes());
}

/// Read all bytes from `fd` into a Vec.
fn read_all(fd: u32) -> Vec<u8> {
    let mut data = Vec::new();
    let mut buf = alloc::vec![0u8; 4096];
    loop {
        match vfs_read(fd, &mut buf) {
            Ok(0) => break,
            Ok(n) => data.extend_from_slice(&buf[..n]),
            Err(_) => break,
        }
    }
    data
}

/// Search `data` for lines containing `pattern`. Print matching lines to
/// stdout (fd 1). Returns the number of matching lines found.
///
/// Options:
///   invert  – print lines that do NOT match (`-v`)
///   ignore_case – case-insensitive comparison (`-i`)
///   line_number – prefix each matching line with its 1-based line number (`-n`)
///   show_filename – prefix each line with `filename:` (used with multiple files)
fn grep_data(
    data: &[u8],
    pattern: &str,
    invert: bool,
    ignore_case: bool,
    line_number: bool,
    show_filename: bool,
    filename: &str,
) -> usize {
    let pat: String = if ignore_case {
        pattern.to_lowercase()
    } else {
        String::from(pattern)
    };

    let mut matches = 0usize;
    let mut lineno = 0usize;
    for line_bytes in data.split(|&b| b == b'\n') {
        lineno += 1;
        let line = core::str::from_utf8(line_bytes).unwrap_or("");
        let haystack: String = if ignore_case {
            line.to_lowercase()
        } else {
            String::from(line)
        };
        let matched = haystack.contains(pat.as_str());
        let print_it = matched != invert;
        if print_it {
            matches += 1;
            let mut out = String::new();
            if show_filename {
                out.push_str(filename);
                out.push(':');
            }
            if line_number {
                out.push_str(&alloc::format!("{}:", lineno));
            }
            out.push_str(line);
            out.push('\n');
            write_str(1, &out);
        }
    }
    matches
}

/// Grep a single open file descriptor. Returns match count.
fn grep_fd(
    fd: u32,
    pattern: &str,
    invert: bool,
    ignore_case: bool,
    line_number: bool,
    show_filename: bool,
    filename: &str,
) -> usize {
    let data = read_all(fd);
    grep_data(
        &data,
        pattern,
        invert,
        ignore_case,
        line_number,
        show_filename,
        filename,
    )
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();

    let mut invert = false;
    let mut ignore_case = false;
    let mut line_number = false;
    let mut count_only = false;
    let mut file_args: Vec<usize> = Vec::new(); // indices into args of file paths
    let mut pattern_idx: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg.starts_with('-') && arg.len() > 1 && arg != "--" {
            for ch in arg[1..].chars() {
                match ch {
                    'v' => invert = true,
                    'i' => ignore_case = true,
                    'n' => line_number = true,
                    'c' => count_only = true,
                    _ => {
                        let msg = alloc::format!("grep: invalid option -- '{}'\n", ch);
                        write_str(2, &msg);
                        stem::syscall::exit(1);
                    }
                }
            }
        } else if pattern_idx.is_none() {
            pattern_idx = Some(i);
        } else {
            file_args.push(i);
        }
        i += 1;
    }

    let pattern_idx = match pattern_idx {
        Some(idx) => idx,
        None => {
            write_str(2, "grep: missing pattern\nusage: grep [-v] [-i] [-n] [-c] pattern [file ...]\n");
            stem::syscall::exit(1);
        }
    };

    let pattern = args[pattern_idx].as_str();
    let show_filename = file_args.len() > 1;
    let mut total_matches = 0usize;

    if file_args.is_empty() {
        // Read from stdin
        let matches = grep_fd(0, pattern, invert, ignore_case, line_number, false, "");
        if count_only {
            write_str(1, &alloc::format!("{}\n", matches));
        }
        total_matches += matches;
    } else {
        for &fidx in &file_args {
            let path = args[fidx].as_str();
            match vfs_open(path, vfs_flags::O_RDONLY) {
                Ok(fd) => {
                    let matches = grep_fd(
                        fd,
                        pattern,
                        invert,
                        ignore_case,
                        line_number,
                        show_filename,
                        path,
                    );
                    let _ = vfs_close(fd);
                    if count_only {
                        if show_filename {
                            write_str(1, &alloc::format!("{}:{}\n", path, matches));
                        } else {
                            write_str(1, &alloc::format!("{}\n", matches));
                        }
                    }
                    total_matches += matches;
                }
                Err(_) => {
                    let msg = alloc::format!("grep: {}: No such file or directory\n", path);
                    write_str(2, &msg);
                }
            }
        }
    }

    // Exit 0 if any match was found, 1 if no match.
    if total_matches > 0 {
        stem::syscall::exit(0)
    } else {
        stem::syscall::exit(1)
    }
}
