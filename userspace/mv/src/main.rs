#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use stem::syscall::{argv_get, exit, vfs_rename, vfs_write};

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

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();
    if args.len() < 2 {
        print("usage: mv <src> <dst>\n");
        exit(1)
    }
    let src = &args[0];
    let dst = &args[1];
    if let Err(e) = vfs_rename(src, dst) {
        print(&alloc::format!("mv: cannot rename '{}' to '{}': {:?}\n", src, dst, e));
        exit(1)
    }
    exit(0)
}
