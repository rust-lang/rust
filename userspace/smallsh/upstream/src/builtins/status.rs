#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use std::path::PathBuf;
use crate::process_pool::ProcessPool;
pub fn status(cwd: &PathBuf, pool: &ProcessPool) {
    println!("CWD: {}", cwd.to_str().unwrap());
    println!("Pool has {} living processes.", pool.len());
    match pool.last_exit_code() {
        Some(code) => println!("Last process exited with code: {}", code),
        None => println!("No process has been run yet."),
    }
}
