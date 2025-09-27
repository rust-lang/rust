// Tests `inherit_handles` by spawning a child process and checking the handle
// count of the child process to be 0.

//@ run-pass
//@ only-windows
//@ needs-subprocess
//@ edition: 2021

#![feature(windows_process_extensions_inherit_handles)]

use std::process::{Child, Command};
use std::{env, mem, ptr, thread, time};

fn main() {
    if std::env::args().skip(1).any(|s| s == "--child") {
        child();
    } else {
        parent();
    }
}

fn parent() {
    let this_exe = std::env::current_exe().unwrap();

    let child_proc = Command::new(&this_exe).arg("--child").inherit_handles(false).spawn().unwrap();

    let mut handle_count = 0;
    unsafe {
        GetProcessHandleCount(child_proc.as_raw_handle(), &mut handle_count);
    }

    assert_eq!(handle_count, 0);
}

// A process that stays running until killed.
fn child() {
    // Don't wait forever if something goes wrong.
    thread::sleep(time::Duration::from_secs(60));
}

// Windows API
mod winapi {
    use std::os::windows::raw::HANDLE;

    #[link(name = "kernel32")]
    extern "system" {
        pub fn GetProcessHandleCount(
            hprocess: HANDLE,
            pdwhandlecount: *mut u32,
        ) -> i32
    }
}
use winapi::*;
