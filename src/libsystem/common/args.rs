// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Global storage for command line arguments
//!
//! The current incarnation of the Rust runtime expects for
//! the processes `argc` and `argv` arguments to be stored
//! in a globally-accessible location for use by the `os` module.
//!
//! Only valid to call on Linux. Mac and Windows use syscalls to
//! discover the command line arguments.
//!
//! FIXME #7756: Would be nice for this to not exist.

#![allow(dead_code)] // different code on OSX/linux/etc

use collections::Vec;

/// One-time global initialization.
pub unsafe fn init(argc: isize, argv: *const *const u8) { imp::init(argc, argv) }

/// One-time global cleanup.
pub unsafe fn cleanup() { imp::cleanup() }

/// Make a clone of the global arguments.
pub fn clone() -> Option<Vec<Vec<u8>>> { imp::clone() }

#[cfg(any(target_os = "linux",
          target_os = "android",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "netbsd",
          target_os = "openbsd"))]
mod imp {
    use libc::c_char;
    use core::mem;
    use c_str::CStr;
    use alloc::boxed::Box;
    use collections::Vec;

    use sync::prelude::*;

    static mut GLOBAL_ARGS_PTR: usize = 0;
    static LOCK: Mutex = Mutex::new();

    pub unsafe fn init(argc: isize, argv: *const *const u8) {
        let args = (0..argc).map(|i| {
            CStr::from_ptr(*argv.offset(i) as *const c_char).to_bytes().to_vec()
        }).collect();

        LOCK.lock();
        let ptr = get_global_ptr();
        let was_none = (*ptr).is_none();
        (*ptr) = Some(Box::new(args));
        LOCK.unlock();

        assert!(was_none);
    }

    pub unsafe fn cleanup() {
        LOCK.lock();
        *get_global_ptr() = None;
        LOCK.unlock();
    }

    pub fn clone() -> Option<Vec<Vec<u8>>> {
        unsafe {
            LOCK.lock();
            let ptr = get_global_ptr();
            let v = (*ptr).as_ref().map(|s| (**s).clone());
            LOCK.unlock();
            v
        }
    }

    fn get_global_ptr() -> *mut Option<Box<Vec<Vec<u8>>>> {
        unsafe { mem::transmute(&GLOBAL_ARGS_PTR) }
    }

}

#[cfg(any(target_os = "macos",
          target_os = "ios",
          target_os = "windows"))]
mod imp {
    use vec::Vec;

    pub unsafe fn init(_argc: isize, _argv: *const *const u8) {
    }

    pub fn cleanup() {
    }

    pub fn clone() -> Option<Vec<Vec<u8>>> {
        panic!()
    }
}
