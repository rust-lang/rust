#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use alloc::string::String;
use alloc::vec::Vec;
use stem::syscall::{env_list, exit, vfs_write};

fn print(msg: &str) {
    let _ = vfs_write(1, msg.as_bytes());
}

fn print_err(msg: &str) {
    let _ = vfs_write(2, msg.as_bytes());
}

/// Parse the blob returned by `env_list` into a Vec of (key, value) pairs.
///
/// Wire format (from kernel sys_env_list):
///   count: u32 LE
///   for each entry:
///     key_len: u32 LE, key bytes (UTF-8)
///     val_len: u32 LE, val bytes (UTF-8)
fn parse_env_blob(blob: &[u8]) -> Vec<(String, String)> {
    let mut result = Vec::new();
    if blob.len() < 4 {
        return result;
    }
    let count = u32::from_le_bytes(blob[0..4].try_into().unwrap_or([0; 4])) as usize;
    let mut offset = 4usize;

    for _ in 0..count {
        if offset + 4 > blob.len() {
            break;
        }
        let key_len =
            u32::from_le_bytes(blob[offset..offset + 4].try_into().unwrap_or([0; 4])) as usize;
        offset += 4;
        if offset + key_len > blob.len() {
            break;
        }
        let key =
            core::str::from_utf8(&blob[offset..offset + key_len]).unwrap_or("").to_string();
        offset += key_len;

        if offset + 4 > blob.len() {
            break;
        }
        let val_len =
            u32::from_le_bytes(blob[offset..offset + 4].try_into().unwrap_or([0; 4])) as usize;
        offset += 4;
        if offset + val_len > blob.len() {
            break;
        }
        let val =
            core::str::from_utf8(&blob[offset..offset + val_len]).unwrap_or("").to_string();
        offset += val_len;

        result.push((key, val));
    }
    result
}

#[stem::main]
fn main(_arg: usize) -> ! {
    // First call: determine needed buffer size
    let needed = match env_list(&mut []) {
        Ok(n) => n,
        Err(_) => {
            print_err("env: failed to list environment\n");
            exit(1)
        }
    };

    if needed == 0 {
        exit(0)
    }

    let mut buf = alloc::vec![0u8; needed];
    match env_list(&mut buf) {
        Ok(_) => {}
        Err(_) => {
            print_err("env: failed to list environment\n");
            exit(1)
        }
    }

    let vars = parse_env_blob(&buf);
    for (key, val) in &vars {
        let line = alloc::format!("{}={}\n", key, val);
        print(&line);
    }

    exit(0)
}
