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

#![crate_name = "rlibc"]
#![experimental]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/master/")]

#![feature(import_shadowing, intrinsics, phase)]
#![no_std]

// This library defines the builtin functions, so it would be a shame for
// LLVM to optimize these function calls to themselves!
#![no_builtins]

#[phase(plugin, link)] extern crate core;

#[cfg(test)] extern crate native;
#[cfg(test)] extern crate test;
#[cfg(test)] extern crate debug;

#[cfg(test)] #[phase(plugin, link)] extern crate std;

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
            return a as i32 - b as i32
        }
        i += 1;
    }
    return 0;
}

#[cfg(test)]
mod test {
    use core::collections::Collection;
    use core::str::StrSlice;
    use core::slice::{MutableSlice, ImmutableSlice};

    use super::{memcmp, memset, memcpy, memmove};

    #[test]
    fn memcmp_single_byte_pointers() {
        unsafe {
            assert_eq!(memcmp(&0xFAu8, &0xFAu8, 1), 0x00);
            assert!(memcmp(&0xEFu8, &0xFEu8, 1) < 0x00);
        }
    }

    #[test]
    fn memcmp_strings() {
        {
            let (x, z) = ("Hello!", "Good Bye.");
            let l = x.len();
            unsafe {
                assert_eq!(memcmp(x.as_ptr(), x.as_ptr(), l), 0);
                assert!(memcmp(x.as_ptr(), z.as_ptr(), l) > 0);
                assert!(memcmp(z.as_ptr(), x.as_ptr(), l) < 0);
            }
        }
        {
            let (x, z) = ("hey!", "hey.");
            let l = x.len();
            unsafe {
                assert!(memcmp(x.as_ptr(), z.as_ptr(), l) < 0);
            }
        }
    }

    #[test]
    fn memset_single_byte_pointers() {
        let mut x: u8 = 0xFF;
        unsafe {
            memset(&mut x, 0xAA, 1);
            assert_eq!(x, 0xAA);
            memset(&mut x, 0x00, 1);
            assert_eq!(x, 0x00);
            x = 0x01;
            memset(&mut x, 0x12, 0);
            assert_eq!(x, 0x01);
        }
    }

    #[test]
    fn memset_array() {
        let mut buffer = [b'X', .. 100];
        unsafe {
            memset(buffer.as_mut_ptr(), b'#' as i32, buffer.len());
        }
        for byte in buffer.iter() { assert_eq!(*byte, b'#'); }
    }

    #[test]
    fn memcpy_and_memcmp_arrays() {
        let (src, mut dst) = ([b'X', .. 100], [b'Y', .. 100]);
        unsafe {
            assert!(memcmp(src.as_ptr(), dst.as_ptr(), 100) != 0);
            let _ = memcpy(dst.as_mut_ptr(), src.as_ptr(), 100);
            assert_eq!(memcmp(src.as_ptr(), dst.as_ptr(), 100), 0);
        }
    }

    #[test]
    fn memmove_overlapping() {
        {
            let mut buffer = [ b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9' ];
            unsafe {
                memmove(&mut buffer[4], &buffer[0], 6);
                let mut i = 0;
                for byte in b"0123012345".iter() {
                    assert_eq!(buffer[i], *byte);
                    i += 1;
                }
            }
        }
        {
            let mut buffer = [ b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9' ];
            unsafe {
                memmove(&mut buffer[0], &buffer[4], 6);
                let mut i = 0;
                for byte in b"4567896789".iter() {
                    assert_eq!(buffer[i], *byte);
                    i += 1;
                }
            }
        }
    }
}
