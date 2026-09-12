// Tests `desktop` by creating a new desktop, spawning a child process onto it
// and checking that the child reports back the expected desktop name.

//@ run-pass
//@ only-windows
//@ needs-subprocess
//@ edition: 2024

#![feature(windows_process_extensions_desktop)]

use std::os::windows::process::CommandExt;
use std::process::{Command, Stdio};
use std::{env, io, process};

fn main() {
    if env::args().skip(1).any(|s| s == "--child") {
        child();
    } else {
        parent();
    }
}

fn parent() {
    let exe = env::current_exe().unwrap();

    // Create a uniquely named desktop on the current window station and keep the
    // handle alive so the desktop is not destroyed while the child runs.
    let desktop_name = format!("rust-test-desktop-{}", process::id());
    let desktop_name_wide: Vec<u16> = desktop_name.encode_utf16().chain([0]).collect();
    let hdesk = unsafe {
        CreateDesktopW(
            desktop_name_wide.as_ptr(),
            core::ptr::null(),
            core::ptr::null(),
            0,
            GENERIC_ALL,
            core::ptr::null(),
        )
    };
    assert!(!hdesk.is_null(), "CreateDesktopW failed: {:?}", io::Error::last_os_error());

    // Spawning with `.desktop` should place the child on our new desktop.
    let output = Command::new(&exe)
        .arg("--child")
        .desktop(&desktop_name)
        .stdout(Stdio::piped())
        .output()
        .unwrap();
    assert!(output.status.success(), "child failed: {:?}", output);
    let reported = String::from_utf8(output.stdout).unwrap();
    assert!(
        reported.trim().eq_ignore_ascii_case(&desktop_name),
        "child ran on unexpected desktop: expected {:?}, got {:?}",
        desktop_name,
        reported.trim(),
    );

    // Without `.desktop` the child inherits the parent's desktop, which is
    // not the one we just created.
    let output = Command::new(&exe).arg("--child").stdout(Stdio::piped()).output().unwrap();
    assert!(output.status.success(), "child failed: {:?}", output);
    let reported = String::from_utf8(output.stdout).unwrap();
    assert!(
        !reported.trim().eq_ignore_ascii_case(&desktop_name),
        "child unexpectedly ran on the created desktop {:?} without being asked to",
        desktop_name,
    );

    unsafe { CloseDesktop(hdesk) };
}

/// Prints the name of the desktop the current process is running on.
fn child() {
    let hdesk = unsafe { GetThreadDesktop(GetCurrentThreadId()) };
    assert!(!hdesk.is_null(), "GetThreadDesktop failed: {:?}", io::Error::last_os_error());

    let mut buffer = [0u16; 256];
    let mut needed = 0u32;
    let ret = unsafe {
        GetUserObjectInformationW(
            hdesk,
            UOI_NAME,
            buffer.as_mut_ptr().cast(),
            size_of_val(&buffer) as u32,
            &raw mut needed,
        )
    };
    assert_ne!(ret, 0, "GetUserObjectInformationW failed: {:?}", io::Error::last_os_error());

    let len = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
    let name = String::from_utf16(&buffer[..len]).unwrap();
    print!("{name}");
}

// Windows API
mod winapi {
    use std::ffi::c_void;
    use std::os::windows::raw::HANDLE;

    pub const GENERIC_ALL: u32 = 0x10000000;
    pub const UOI_NAME: i32 = 2;

    #[link(name = "user32")]
    unsafe extern "system" {
        pub fn CreateDesktopW(
            lpszDesktop: *const u16,
            lpszDevice: *const u16,
            pDevmode: *const c_void,
            dwFlags: u32,
            dwDesiredAccess: u32,
            lpsa: *const c_void,
        ) -> HANDLE;
        pub fn CloseDesktop(hDesktop: HANDLE) -> i32;
        pub fn GetThreadDesktop(dwThreadId: u32) -> HANDLE;
        pub fn GetUserObjectInformationW(
            hObj: HANDLE,
            nIndex: i32,
            pvInfo: *mut c_void,
            nLength: u32,
            lpnLengthNeeded: *mut u32,
        ) -> i32;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn GetCurrentThreadId() -> u32;
    }
}
use winapi::*;
