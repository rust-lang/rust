// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// compile-flags:--test

// NB: These tests kill child processes. Valgrind sees these children as leaking
// memory, which makes for some *confusing* logs. That's why these are here
// instead of in std.

use std::libc;
use std::run;
use std::str;
use std::io;

#[test]
fn test_destroy_once() {
    #[cfg(not(target_os="android"))]
    static PROG: &'static str = "echo";
    #[cfg(target_os="android")]
    static PROG: &'static str = "ls"; // android don't have echo binary

    let mut p = run::Process::new(PROG, [], run::ProcessOptions::new());
    p.destroy(); // this shouldn't crash (and nor should the destructor)
}

#[test]
fn test_destroy_twice() {
    #[cfg(not(target_os="android"))]
    static PROG: &'static str = "echo";
    #[cfg(target_os="android")]
    static PROG: &'static str = "ls"; // android don't have echo binary

    let mut p = run::Process::new(PROG, [], run::ProcessOptions::new());
    p.destroy(); // this shouldnt crash...
    io::io_error::cond.trap(|_| {}).inside(|| {
        p.destroy(); // ...and nor should this (and nor should the destructor)
    })
}

fn test_destroy_actually_kills(force: bool) {

    #[cfg(unix,not(target_os="android"))]
    static BLOCK_COMMAND: &'static str = "cat";

    #[cfg(unix,target_os="android")]
    static BLOCK_COMMAND: &'static str = "/system/bin/cat";

    #[cfg(windows)]
    static BLOCK_COMMAND: &'static str = "cmd";

    #[cfg(unix,not(target_os="android"))]
    fn process_exists(pid: libc::pid_t) -> bool {
        let run::ProcessOutput {output, ..} = run::process_output("ps", [~"-p", pid.to_str()]);
        str::from_utf8(output).contains(pid.to_str())
    }

    #[cfg(unix,target_os="android")]
    fn process_exists(pid: libc::pid_t) -> bool {
        let run::ProcessOutput {output, ..} = run::process_output("/system/bin/ps", [pid.to_str()]);
        str::from_utf8(output).contains(~"root")
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
    let mut p = run::Process::new(BLOCK_COMMAND, [], run::ProcessOptions::new());

    assert!(process_exists(p.get_id()));

    if force {
        p.force_destroy();
    } else {
        p.destroy();
    }

    assert!(!process_exists(p.get_id()));
}

#[test]
fn test_unforced_destroy_actually_kills() {
    test_destroy_actually_kills(false);
}

#[test]
fn test_forced_destroy_actually_kills() {
    test_destroy_actually_kills(true);
}
