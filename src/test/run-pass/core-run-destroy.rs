// Copyright 2012-2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// compile-flags:--test

// NB: These tests kill child processes. Valgrind sees these children as leaking
// memory, which makes for some *confusing* logs. That's why these are here
// instead of in std.

use std::io::timer;
use std::libc;
use std::str;
use std::io::process::{Process, ProcessOutput};

#[test]
fn test_destroy_once() {
    #[cfg(not(target_os="android"))]
    static mut PROG: &'static str = "echo";

    #[cfg(target_os="android")]
    static mut PROG: &'static str = "ls"; // android don't have echo binary

    let mut p = unsafe {Process::new(PROG, []).unwrap()};
    p.signal_exit().unwrap(); // this shouldn't crash (and nor should the destructor)
}

#[test]
fn test_destroy_twice() {
    #[cfg(not(target_os="android"))]
    static mut PROG: &'static str = "echo";
    #[cfg(target_os="android")]
    static mut PROG: &'static str = "ls"; // android don't have echo binary

    let mut p = match unsafe{Process::new(PROG, [])} {
        Ok(p) => p,
        Err(e) => fail!("wut: {}", e),
    };
    p.signal_exit().unwrap(); // this shouldnt crash...
    p.signal_exit().unwrap(); // ...and nor should this (and nor should the destructor)
}

fn test_destroy_actually_kills(force: bool) {

    #[cfg(unix,not(target_os="android"))]
    static mut BLOCK_COMMAND: &'static str = "cat";

    #[cfg(unix,target_os="android")]
    static mut BLOCK_COMMAND: &'static str = "/system/bin/cat";

    #[cfg(windows)]
    static mut BLOCK_COMMAND: &'static str = "cmd";

    #[cfg(unix,not(target_os="android"))]
    fn process_exists(pid: libc::pid_t) -> bool {
        let ProcessOutput {output, ..} = Process::output("ps", [~"-p", pid.to_str()])
            .unwrap();
        str::from_utf8_owned(output).unwrap().contains(pid.to_str())
    }

    #[cfg(unix,target_os="android")]
    fn process_exists(pid: libc::pid_t) -> bool {
        let ProcessOutput {output, ..} = Process::output("/system/bin/ps", [pid.to_str()])
            .unwrap();
        str::from_utf8_owned(output).unwrap().contains(~"root")
    }

    #[cfg(windows)]
    fn process_exists(pid: libc::pid_t) -> bool {
        use std::libc::types::os::arch::extra::DWORD;
        use std::libc::funcs::extra::kernel32::{CloseHandle, GetExitCodeProcess, OpenProcess};
        use std::libc::consts::os::extra::{FALSE, PROCESS_QUERY_INFORMATION, STILL_ACTIVE };

        unsafe {
            let process = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, pid as DWORD);
            if process.is_null() {
                return false;
            }
            // process will be non-null if the process is alive, or if it died recently
            let mut status = 0;
            GetExitCodeProcess(process, &mut status);
            CloseHandle(process);
            return status == STILL_ACTIVE;
        }
    }

    // this process will stay alive indefinitely trying to read from stdin
    let mut p = unsafe {Process::new(BLOCK_COMMAND, []).unwrap()};

    assert!(process_exists(p.id()));

    if force {
        p.signal_kill().unwrap();
    } else {
        p.signal_exit().unwrap();
    }

    if process_exists(p.id()) {
        timer::sleep(500);
        assert!(!process_exists(p.id()));
    }
}

#[test]
fn test_unforced_destroy_actually_kills() {
    test_destroy_actually_kills(false);
}

#[test]
fn test_forced_destroy_actually_kills() {
    test_destroy_actually_kills(true);
}
