#![cfg(any(target_family = "unix", target_family = "windows"))]

use std::fs::File;
use std::io::{Read, Write};

mod common;

#[cfg(unix)]
fn switch_stdout_to(file: File) {
    use std::os::unix::prelude::*;

    extern "C" {
        fn dup2(old: i32, new: i32) -> i32;
    }

    unsafe {
        assert_eq!(dup2(file.as_raw_fd(), 1), 1);
    }
}

#[cfg(windows)]
fn switch_stdout_to(file: File) {
    use std::os::windows::prelude::*;

    extern "system" {
        fn SetStdHandle(nStdHandle: u32, handle: *mut u8) -> i32;
    }

    const STD_OUTPUT_HANDLE: u32 = (-11i32) as u32;

    unsafe {
        let rc = SetStdHandle(STD_OUTPUT_HANDLE, file.into_raw_handle() as *mut _);
        assert!(rc != 0);
    }
}

#[test]
fn switch_stdout() {
    let temp = common::tmpdir();
    let path = temp.join("switch-stdout-output");
    let f = File::create(&path).unwrap();

    let mut stdout = std::io::stdout();
    stdout.write(b"foo\n").unwrap();
    stdout.flush().unwrap();
    switch_stdout_to(f);
    stdout.write(b"bar\n").unwrap();
    stdout.flush().unwrap();

    let mut contents = String::new();
    File::open(&path).unwrap().read_to_string(&mut contents).unwrap();
    assert_eq!(contents, "bar\n");
}
