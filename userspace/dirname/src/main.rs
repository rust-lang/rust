#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use alloc::string::String;
use alloc::vec::Vec;
use stem::syscall::{argv_get, exit, vfs_write};

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

    stem::utils::parse_argv(&buf)
        .into_iter()
        .skip(1)
        .filter_map(|b| core::str::from_utf8(b).ok().map(String::from))
        .collect()
}

fn print(msg: &str) {
    let _ = vfs_write(1, msg.as_bytes());
}

fn trim_trailing_slashes(path: &str) -> &str {
    if path.is_empty() {
        return path;
    }

    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() { "/" } else { trimmed }
}

fn dirname(path: &str) -> &str {
    if path.is_empty() {
        return ".";
    }

    let trimmed = trim_trailing_slashes(path);
    if trimmed == "/" {
        return "/";
    }

    match trimmed.rfind('/') {
        Some(0) => "/",
        Some(idx) => trim_trailing_slashes(&trimmed[..idx]),
        None => ".",
    }
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();
    if args.len() != 1 {
        print("usage: dirname <path>\n");
        exit(1)
    }

    print(dirname(&args[0]));
    print("\n");
    exit(0)
}
