// run-pass
#![allow(dead_code)]

#![feature(rustc_private)]

extern crate libc;

type DWORD = u32;
type HANDLE = *mut u8;
type BOOL = i32;

#[cfg(windows)]
extern "system" {
    fn SetStdHandle(nStdHandle: DWORD, nHandle: HANDLE) -> BOOL;
}

#[cfg(windows)]
fn close_stdout() {
    const STD_OUTPUT_HANDLE: DWORD = -11i32 as DWORD;
    unsafe { SetStdHandle(STD_OUTPUT_HANDLE, 0 as HANDLE); }
}

#[cfg(windows)]
fn main() {
    close_stdout();
    println!("hello");
    println!("world");
}

#[cfg(not(windows))]
fn main() {}
