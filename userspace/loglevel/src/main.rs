#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use stem::syscall::{argv_get, log_set_level};
use stem::{info, println};

fn get_args() -> alloc::vec::Vec<alloc::string::String> {
    let mut len = 0;
    if let Ok(l) = argv_get(&mut []) {
        len = l;
    }
    if len == 0 {
        return alloc::vec::Vec::new();
    }
    let mut buf = alloc::vec![0u8; len];
    if argv_get(&mut buf).is_err() {
        return alloc::vec::Vec::new();
    }

    let mut args = alloc::vec::Vec::new();
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
                args.push(alloc::string::String::from(s));
            }
            offset += str_len;
        }
    }
    args
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let args = get_args();
    if args.len() < 2 {
        println!("Usage: loglevel <0-5>");
        println!("  0: Contract Only");
        println!("  1: Error");
        println!("  2: Warn");
        println!("  3: Info");
        println!("  4: Debug (Default)");
        println!("  5: Trace");
        stem::syscall::exit(0);
    }

    let level_str = &args[1];
    let level: u8 = match level_str.as_str() {
        "0" => 0,
        "1" => 1,
        "2" => 2,
        "3" => 3,
        "4" => 4,
        "5" => 5,
        _ => {
            println!("Invalid log level: {}", level_str);
            stem::syscall::exit(1);
        }
    };

    match log_set_level(level) {
        Ok(_) => println!("Log level set to {}", level),
        Err(e) => println!("Failed to set log level: {:?}", e),
    }

    stem::syscall::exit(0);
}
