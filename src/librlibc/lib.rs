// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A bare-metal library supplying functions rustc may lower code to
//!
//! This library is not intended for general use, and is superseded by a system
//! libc if one is available. In a freestanding context, however, common
//! functions such as memset, memcpy, etc are not implemented. This library
//! provides an implementation of these functions which are either required by
//! libcore or called by rustc implicitly.
//!
//! This library is never included by default, and must be manually included if
//! necessary. It is an error to include this library when also linking with
//! the system libc library.

#![crate_id = "rlibc#0.11.0-pre"]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/")]
#![feature(intrinsics)]

#![no_std]
#![experimental]

// This library defines the builtin functions, so it would be a shame for
// LLVM to optimize these function calls to themselves!
#![no_builtins]

#[cfg(test)] extern crate std;
#[cfg(test)] extern crate native;

// Require the offset intrinsics for LLVM to properly optimize the
// implementations below. If pointer arithmetic is done through integers the
// optimizations start to break down.
extern "rust-intrinsic" {
    fn offset<T>(dst: *const T, offset: int) -> *const T;
}

#[no_mangle]
pub unsafe extern "C" fn memcpy(dest: *mut u8, src: *const u8,
                                n: uint) -> *mut u8 {
    let mut i = 0;
    while i < n {
        *(offset(dest as *const u8, i as int) as *mut u8) =
            *offset(src, i as int);
        i += 1;
    }
    return dest;
}

#[no_mangle]
pub unsafe extern "C" fn memmove(dest: *mut u8, src: *const u8,
                                 n: uint) -> *mut u8 {
    if src < dest as *const u8 { // copy from end
        let mut i = n;
        while i != 0 {
            i -= 1;
            *(offset(dest as *const u8, i as int) as *mut u8) =
                *offset(src, i as int);
        }
    } else { // copy from beginning
        let mut i = 0;
        while i < n {
            *(offset(dest as *const u8, i as int) as *mut u8) =
                *offset(src, i as int);
            i += 1;
        }
    }
    return dest;
}

#[no_mangle]
pub unsafe extern "C" fn memset(s: *mut u8, c: i32, n: uint) -> *mut u8 {
    let mut i = 0;
    while i < n {
        *(offset(s as *const u8, i as int) as *mut u8) = c as u8;
        i += 1;
    }
    return s;
}

#[no_mangle]
pub unsafe extern "C" fn memcmp(s1: *const u8, s2: *const u8, n: uint) -> i32 {
    let mut i = 0;
    while i < n {
        let a = *offset(s1, i as int);
        let b = *offset(s2, i as int);
        if a != b {
            return (a - b) as i32
        }
        i += 1;
    }
    return 0;
}

#[test] fn work_on_windows() { } // FIXME #10872 needed for a happy windows
