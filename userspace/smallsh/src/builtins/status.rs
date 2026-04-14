#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use std::path::Path;

use crate::process_pool::ProcessPool;

pub fn status(cwd: &Path, pool: &ProcessPool) {
    println!("CWD: {}", cwd.display());
    println!("Pool has {} living processes.", pool.len());
    match pool.last_exit_code() {
        Some(code) => println!("Last process exited with code: {}", code),
        None => println!("No process has been run yet."),
    }
}
