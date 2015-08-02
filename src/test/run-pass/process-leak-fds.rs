// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(convert)]
#![feature(libc)]
#![feature(process_leak_fds)]

extern crate libc;

use std::collections::HashSet;
use std::fs::{File, read_dir};
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::str::FromStr;

#[cfg(any(target_os = "linux",
          target_os = "android"))]
fn check_fds(mut whitelist: HashSet<RawFd>) -> ! {
    match read_dir("/dev/fd") {
        Ok(dir) => {
            let current = dir.as_raw_fd();
            for ret in dir {
                match ret {
                    Ok(entry) => {
                        let filename = entry.file_name();
                        let filename = match filename.to_str() {
                            Some(s) => s,
                            None => panic!("Failed to convert {:?}", filename),
                        };
                        let fd = match FromStr::from_str(filename) {
                            Ok(fd) => fd,
                            Err(e) => panic!("Failed to convert {:?}: {}", filename, e),
                        };
                        if fd != current && !whitelist.contains(&fd) {
                            panic!("Unexpected leaked FD {}", fd);
                        }
                        whitelist.remove(&fd);
                    }
                    Err(e) => panic!("Failed to get directory entry: {}", e),
                }
            }
        }
        Err(e) => panic!("Failed to get directory entry: {}", e),
    }
    if whitelist.len() != 0 {
        panic!("Failed to leak FDs: {:?}", whitelist);
    }
    std::process::exit(0);
}

macro_rules! add_args {
    ($cmd: expr, $whitelist: expr) => {
        $cmd.args($whitelist.iter().map(|x| format!("{}", x)).collect::<Vec<_>>().as_slice())
    }
}

#[cfg(any(target_os = "linux",
          target_os = "android"))]
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() > 1 {
        let whitelist = args[1..].iter().map(|x| FromStr::from_str(x).unwrap()).collect();
        check_fds(whitelist);
    } else {
        let exe = std::env::current_exe().unwrap();
        // Preload stdio
        let mut whitelist = HashSet::new();
        for fd in &[libc::STDIN_FILENO, libc::STDOUT_FILENO, libc::STDERR_FILENO] {
            whitelist.insert(*fd);
        }

        let mut cmd = Command::new(exe.clone());
        let ret = add_args!(cmd, whitelist).
            status().unwrap().code().unwrap();
        if ret != 0 {
            panic!("Test #1 failed");
        }

        // Leak a first file descriptor
        let fd1 = unsafe { libc::open(exe.to_str().unwrap().as_ptr() as *const i8,
                                      libc::O_RDONLY, 0) };

        // Launch a command without FD leak
        let mut cmd = Command::new(exe.clone());
        let ret = add_args!(cmd, whitelist).
            leak_fds(false).
            status().unwrap().code().unwrap();
        if ret != 0 {
            panic!("Test #2 failed");
        }

        // Launch a command with the first FD leak
        whitelist.insert(fd1);
        let mut cmd = Command::new(exe.clone());
        let ret = add_args!(cmd, whitelist).
            leak_fds_whitelist(whitelist.clone()).
            status().unwrap().code().unwrap();
        if ret != 0 {
            panic!("Test #3 failed");
        }

        // Leak a second file descriptor
        let _ = unsafe { libc::open(exe.to_str().unwrap().as_ptr() as *const i8,
                                    libc::O_RDONLY, 0) };

        // Open a third file descriptor but with O_CLOEXEC
        let fd3 = File::open(&exe).unwrap();

        // Launch a command with the first FD leak (but not the second nor the third)
        let mut cmd = Command::new(exe.clone());
        let ret = add_args!(cmd, whitelist).
            leak_fds_whitelist(whitelist.clone()).
            status().unwrap().code().unwrap();
        if ret != 0 {
            panic!("Test #4 failed");
        }

        // Launch a command with the first and the third FD (but not the second)
        whitelist.insert(fd3.as_raw_fd());
        let mut cmd = Command::new(exe.clone());
        let ret = add_args!(cmd, whitelist).
            leak_fds_whitelist(whitelist.clone()).
            status().unwrap().code().unwrap();
        if ret != 0 {
            panic!("Test #5 failed");
        }
    }
}

#[cfg(any(target_os = "bitrig",
          target_os = "dragonfly",
          target_os = "freebsd",
          target_os = "macos",
          target_os = "netbsd",
          target_os = "openbsd",
          target_os = "windows"))]
pub fn main() { }
