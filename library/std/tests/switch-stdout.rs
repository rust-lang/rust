#![cfg(any(target_family = "unix", target_family = "windows"))]

use std::fs::File;
use std::io::{Read, Write};

mod common;

#[cfg(unix)]
use std::os::fd::OwnedFd;
#[cfg(windows)]
use std::os::windows::io::OwnedHandle;

#[cfg(unix)]
fn switch_stdout_to(file: OwnedFd) -> OwnedFd {
    use std::os::unix::prelude::*;

    extern "C" {
        fn dup(old: i32) -> i32;
        fn dup2(old: i32, new: i32) -> i32;
    }

    unsafe {
        let orig_fd = dup(1);
        assert_ne!(orig_fd, -1);
        let res = OwnedFd::from_raw_fd(orig_fd);
        assert_eq!(dup2(file.as_raw_fd(), 1), 1);
        res
    }
}

#[cfg(windows)]
fn switch_stdout_to(file: OwnedHandle) -> OwnedHandle {
    use std::os::windows::prelude::*;

    extern "system" {
        fn GetStdHandle(nStdHandle: u32) -> *mut u8;
        fn SetStdHandle(nStdHandle: u32, handle: *mut u8) -> i32;
    }

    const STD_OUTPUT_HANDLE: u32 = (-11i32) as u32;
    const INVALID_HANDLE_VALUE: *mut u8 = !0 as *mut u8;

    unsafe {
        let orig_hdl = GetStdHandle(STD_OUTPUT_HANDLE);
        assert!(!orig_hdl.is_null() && orig_hdl != INVALID_HANDLE_VALUE);
        let rc = SetStdHandle(STD_OUTPUT_HANDLE, file.into_raw_handle() as *mut _);
        assert!(rc != 0);
        OwnedHandle::from_raw_handle(orig_hdl as _)
    }
}

#[test]
#[cfg_attr(miri, ignore)] // dup/SetStdHandle not supported by Miri
fn switch_stdout() {
    let temp = common::tmpdir();
    let path = temp.join("switch-stdout-output");
    let f = File::create(&path).unwrap();

    let mut stdout = std::io::stdout();
    stdout.write(b"foo\n").unwrap();
    stdout.flush().unwrap();
    let orig_hdl = switch_stdout_to(f.into());
    stdout.write(b"bar\n").unwrap();
    stdout.flush().unwrap();

    switch_stdout_to(orig_hdl);

    let mut contents = String::new();
    File::open(&path).unwrap().read_to_string(&mut contents).unwrap();
    assert_eq!(contents, "bar\n");
}
