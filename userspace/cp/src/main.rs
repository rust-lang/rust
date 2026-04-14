#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use stem::abi::syscall::vfs_flags;
use stem::syscall::{argv_get, exit, vfs_close, vfs_open, vfs_read, vfs_write};

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

fn copy_file(src: &str, dst: &str) -> Result<(), ()> {
    let in_fd = vfs_open(src, vfs_flags::O_RDONLY).map_err(|_| {
        print(&alloc::format!("cp: cannot open '{}'\n", src));
    })?;

    let out_fd = vfs_open(
        dst,
        vfs_flags::O_WRONLY | vfs_flags::O_CREAT | vfs_flags::O_TRUNC,
    )
    .map_err(|_| {
        let _ = vfs_close(in_fd);
        print(&alloc::format!("cp: cannot create '{}'\n", dst));
    })?;

    let mut buf = alloc::vec![0u8; 32768];
    let result = 'copy: loop {
        match vfs_read(in_fd, &mut buf) {
            Ok(0) => break Ok(()),
            Ok(n) => {
                let mut written = 0;
                while written < n {
                    match vfs_write(out_fd, &buf[written..n]) {
                        Ok(0) => {
                            print(&alloc::format!("cp: write error on '{}'\n", dst));
                            break 'copy Err(());
                        }
                        Ok(nw) => written += nw,
                        Err(_) => {
                            print(&alloc::format!("cp: write error on '{}'\n", dst));
                            break 'copy Err(());
                        }
                    }
                }
            }
            Err(_) => {
                print(&alloc::format!("cp: read error on '{}'\n", src));
                break Err(());
            }
        }
    };

    let _ = vfs_close(in_fd);
    let _ = vfs_close(out_fd);
    result
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();
    if args.len() < 2 {
        print("usage: cp <src> <dst>\n");
        exit(1)
    }
    let src = &args[0];
    let dst = &args[1];
    if copy_file(src, dst).is_err() {
        exit(1)
    }
    exit(0)
}
