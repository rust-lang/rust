//@ run-pass
//@ ignore-android
//@ needs-subprocess

#![feature(rustc_private)]

#[cfg(unix)]
extern crate libc;

use std::env;
use std::ffi::c_int;
use std::io::{self, Read, Write};
use std::process::{Command, Stdio};

#[cfg(unix)]
unsafe fn without_stdio<R, F: FnOnce() -> R>(f: F) -> R {
    let doit = |a| {
        let r = libc::dup(a);
        assert!(r >= 0);
        return r
    };
    let a = doit(0);
    let b = doit(1);
    let c = doit(2);

    assert!(libc::close(0) >= 0);
    assert!(libc::close(1) >= 0);
    assert!(libc::close(2) >= 0);

    let r = f();

    assert!(libc::dup2(a, 0) >= 0);
    assert!(libc::dup2(b, 1) >= 0);
    assert!(libc::dup2(c, 2) >= 0);

    return r
}

#[cfg(unix)]
fn assert_fd_is_valid(fd: c_int) {
    if unsafe { libc::fcntl(fd, libc::F_GETFD) == -1 } {
        panic!("file descriptor {} is not valid: {}", fd, io::Error::last_os_error());
    }
}

#[cfg(windows)]
fn assert_fd_is_valid(_fd: c_int) {}

#[cfg(windows)]
unsafe fn without_stdio<R, F: FnOnce() -> R>(f: F) -> R {
    type DWORD = u32;
    type HANDLE = *mut u8;
    type BOOL = i32;

    const STD_INPUT_HANDLE: DWORD = -10i32 as DWORD;
    const STD_OUTPUT_HANDLE: DWORD = -11i32 as DWORD;
    const STD_ERROR_HANDLE: DWORD = -12i32 as DWORD;
    const INVALID_HANDLE_VALUE: HANDLE = !0 as HANDLE;

    extern "system" {
        fn GetStdHandle(which: DWORD) -> HANDLE;
        fn SetStdHandle(which: DWORD, handle: HANDLE) -> BOOL;
    }

    let doit = |id| {
        let handle = GetStdHandle(id);
        assert!(handle != INVALID_HANDLE_VALUE);
        assert!(SetStdHandle(id, INVALID_HANDLE_VALUE) != 0);
        return handle
    };

    let a = doit(STD_INPUT_HANDLE);
    let b = doit(STD_OUTPUT_HANDLE);
    let c = doit(STD_ERROR_HANDLE);

    let r = f();

    let doit = |id, handle| {
        assert!(SetStdHandle(id, handle) != 0);
    };
    doit(STD_INPUT_HANDLE, a);
    doit(STD_OUTPUT_HANDLE, b);
    doit(STD_ERROR_HANDLE, c);

    return r
}

fn main() {
    if env::args().len() > 1 {
        // Writing to stdout & stderr should not panic.
        println!("test");
        assert!(io::stdout().write(b"test\n").is_ok());
        assert!(io::stderr().write(b"test\n").is_ok());

        // Stdin should be at EOF.
        assert_eq!(io::stdin().read(&mut [0; 10]).unwrap(), 0);

        // Standard file descriptors should be valid on UNIX:
        assert_fd_is_valid(0);
        assert_fd_is_valid(1);
        assert_fd_is_valid(2);
        return
    }

    // First, make sure reads/writes without stdio work if stdio itself is
    // missing.
    let (a, b, c) = unsafe {
        without_stdio(|| {
            let a = io::stdout().write(b"test\n");
            let b = io::stderr().write(b"test\n");
            let c = io::stdin().read(&mut [0; 10]);

            (a, b, c)
        })
    };

    assert_eq!(a.unwrap(), 5);
    assert_eq!(b.unwrap(), 5);
    assert_eq!(c.unwrap(), 0);

    // Second, spawn a child and do some work with "null" descriptors to make
    // sure it's ok
    let me = env::current_exe().unwrap();
    let status = Command::new(&me)
                        .arg("next")
                        .stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status().unwrap();
    assert!(status.success(), "{} isn't a success", status);

    // Finally, close everything then spawn a child to make sure everything is
    // *still* ok.
    let status = unsafe {
        without_stdio(|| Command::new(&me).arg("next").status())
    }.unwrap();
    assert!(status.success(), "{} isn't a success", status);
}
