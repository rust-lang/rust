// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_imports)]

// ignore-windows
// ignore-wasm32-bare no libc to test ffi with

// issue-53200

#![feature(rustc_private)]
extern crate libc;

use std::env;

// FIXME: more platforms?
#[cfg(target_os = "linux")]
fn main() {
    unsafe { libc::clearenv(); }
    assert_eq!(env::vars().count(), 0);
}

#[cfg(not(target_os = "linux"))]
fn main() {}
