// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ignore this test on Android, because it segfaults there.

// ignore-android
// ignore-windows
// no-prefer-dynamic

#![feature(convert)]
#![feature(libc)]

extern crate libc;

use libc::c_char;
use libc::execve;
use std::env;
use std::ffi::OsStr;
use std::ptr;

fn main() {
    if env::args_os().next().is_none() {
        for (key, value) in env::vars_os() {
            panic!("found env value {:?} {:?}", key, value);
        }
        return;
    }

    let current_exe = env::current_exe().unwrap().into_os_string().to_cstring().unwrap();
    let new_env_var = OsStr::new("FOOBAR").to_cstring().unwrap();
    let filename: *const c_char = current_exe.as_ptr();
    let argv: &[*const c_char] = &[ptr::null()];
    let envp: &[*const c_char] = &[new_env_var.as_ptr(), ptr::null()];
    unsafe {
        execve(filename, &argv[0], &envp[0]);
    }
    panic!("execve failed");
}
