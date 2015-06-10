// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(libc)]

extern crate libc;

#[cfg(windows)]
extern "system" {
    fn SetStdHandle(nStdHandle: libc::DWORD, nHandle: libc::HANDLE) -> libc::BOOL;
}

#[cfg(windows)]
fn close_stdout() {
    const STD_OUTPUT_HANDLE: libc::DWORD = -11i32 as libc::DWORD;
    unsafe { SetStdHandle(STD_OUTPUT_HANDLE, 0 as libc::HANDLE); }
}

#[cfg(windows)]
fn main() {
    close_stdout();
    println!("hello world");
}

#[cfg(not(windows))]
fn main() {}
