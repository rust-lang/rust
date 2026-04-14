#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use alloc::string::String;
use alloc::vec::Vec;
use stem::abi::syscall::vfs_flags;
use stem::syscall::{
    argv_get, exit, vfs_close, vfs_open, vfs_read, vfs_seek, vfs_stat, vfs_write,
};

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

fn touch_path(path: &str) -> Result<(), ()> {
    match vfs_open(path, vfs_flags::O_RDWR) {
        Ok(fd) => {
            let stat = match vfs_stat(fd) {
                Ok(stat) => stat,
                Err(_) => {
                    let _ = vfs_close(fd);
                    return Err(());
                }
            };
            let kind = stat.mode & 0o170000;

            // Without a dedicated utime syscall, the closest approximation for a
            // regular non-empty file is to rewrite the first byte unchanged.
            if kind == 0o100000 && stat.size > 0 {
                let mut byte = [0u8; 1];
                if vfs_seek(fd, 0, 0).is_err()
                    || vfs_read(fd, &mut byte).is_err()
                    || vfs_seek(fd, 0, 0).is_err()
                    || vfs_write(fd, &byte).is_err()
                {
                    let _ = vfs_close(fd);
                    return Err(());
                }
            }

            let _ = vfs_close(fd);
            Ok(())
        }
        Err(_) => {
            let fd = vfs_open(path, vfs_flags::O_WRONLY | vfs_flags::O_CREAT).map_err(|_| ())?;
            let _ = vfs_close(fd);
            Ok(())
        }
    }
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();
    if args.is_empty() {
        print("usage: touch <path>...\n");
        exit(1)
    }

    let mut had_error = false;
    for path in &args {
        if touch_path(path).is_err() {
            print(&alloc::format!("touch: cannot touch '{}'\n", path));
            had_error = true;
        }
    }

    if had_error {
        exit(1)
    }
    exit(0)
}
