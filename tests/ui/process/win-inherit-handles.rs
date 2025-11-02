// Tests `inherit_handles` by spawning a child process and checking its handle
// count to be greater than when not setting the option.

//@ run-pass
//@ only-windows
//@ needs-subprocess
//@ edition: 2024

#![feature(windows_process_extensions_inherit_handles)]

use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::Duration;
use std::{env, io, thread};

fn main() {
    if std::env::args().skip(1).any(|s| s == "--child") {
        child();
    } else {
        parent();
    }
}

fn parent() {
    let with_inherit_count = child_handle_count(true);
    let without_inherit_count = child_handle_count(false);
    // Only compare the two values instead of only expecting a hard 1 for
    // robustness, although only 1 has ever been observed here.
    assert!(
        with_inherit_count > without_inherit_count,
        "Child process handle count unexpectedly smaller when inheriting handles compared to when \
        not: {} <= {}",
        with_inherit_count,
        without_inherit_count,
    );
}

/// Spawns the current program as a child process and returns its handle count.
fn child_handle_count(inherit_handles: bool) -> u32 {
    let mut child_proc = Command::new(&env::current_exe().unwrap())
        .arg("--child")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .inherit_handles(inherit_handles)
        .spawn()
        .unwrap();

    let mut handle_count = 0;
    let ret = unsafe { GetProcessHandleCount(child_proc.as_raw_handle(), &raw mut handle_count) };
    assert_ne!(
        ret,
        0,
        "GetProcessHandleCount failed: {:?}",
        io::Error::last_os_error(),
    );

    // Cleanup.
    child_proc.kill().unwrap();
    child_proc.wait().unwrap();

    handle_count
}

/// A process that stays running until killed.
fn child() {
    // Don't wait forever if something goes wrong.
    thread::sleep(Duration::from_secs(10));
}

// Windows API
mod winapi {
    use std::os::windows::raw::HANDLE;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn GetProcessHandleCount(hprocess: HANDLE, pdwhandlecount: *mut u32) -> i32;
    }
}
use winapi::*;
