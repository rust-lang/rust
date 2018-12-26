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
