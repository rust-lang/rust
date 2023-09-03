// Tests `inherit_handles` by spawning a child process and checking the handle
// count of the child process to be 1.

//@ run-pass
//@ only-windows
//@ needs-subprocess
//@ edition: 2024

#![feature(windows_process_extensions_inherit_handles)]

use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::process::{Command, Stdio};
use std::{thread, time};

fn main() {
    if std::env::args().skip(1).any(|s| s == "--child") {
        child();
    } else {
        parent();
    }
}

fn parent() {
    let this_exe = std::env::current_exe().unwrap();

    let mut child_proc = Command::new(&this_exe)
        .arg("--child")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .inherit_handles(false)
        .spawn()
        .unwrap();

    let mut handle_count = 0;
    unsafe {
        GetProcessHandleCount(child_proc.as_raw_handle(), &mut handle_count);
    }

    // Only a single handle to the PE file the process is spawned from is expected at this point.
    assert_eq!(handle_count, 1);

    // Cleanup.
    child_proc.kill().unwrap();
    child_proc.wait().unwrap();
}

// A process that stays running until killed.
fn child() {
    // Don't wait forever if something goes wrong.
    thread::sleep(time::Duration::from_secs(10));
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
