// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Global initialization and retreival of command line arguments.
//!
//! On some platforms these are stored during runtime startup,
//! and on some they are retrieved from the system on demand.

#![allow(dead_code)] // runtime init functions not used during testing

use ffi::OsString;
use marker::PhantomData;
use vec;

/// One-time global initialization.
pub unsafe fn init(argc: isize, argv: *const *const u8) { imp::init(argc, argv) }

/// One-time global cleanup.
pub unsafe fn cleanup() { imp::cleanup() }

/// Returns the command line arguments
pub fn args() -> Args {
    imp::args()
}

pub struct Args {
    iter: vec::IntoIter<OsString>,
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

impl Iterator for Args {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize { self.iter.len() }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> { self.iter.next_back() }
}

mod imp {
    use os::unix::prelude::*;
    use mem;
    use ffi::{CStr, OsString};
    use marker::PhantomData;
    use libc;
    use super::Args;

    use sys_common::mutex::Mutex;

    static mut GLOBAL_ARGS_PTR: usize = 0;
    static LOCK: Mutex = Mutex::new();

    pub unsafe fn init(argc: isize, argv: *const *const u8) {
        let args = (0..argc).map(|i| {
            CStr::from_ptr(*argv.offset(i) as *const libc::c_char).to_bytes().to_vec()
        }).collect();

        LOCK.lock();
        let ptr = get_global_ptr();
        assert!((*ptr).is_none());
        (*ptr) = Some(box args);
        LOCK.unlock();
    }

    pub unsafe fn cleanup() {
        LOCK.lock();
        *get_global_ptr() = None;
        LOCK.unlock();
    }

    pub fn args() -> Args {
        let bytes = clone().unwrap_or(Vec::new());
        let v: Vec<OsString> = bytes.into_iter().map(|v| {
            OsStringExt::from_vec(v)
        }).collect();
        Args { iter: v.into_iter(), _dont_send_or_sync_me: PhantomData }
    }

    fn clone() -> Option<Vec<Vec<u8>>> {
        unsafe {
            LOCK.lock();
            let ptr = get_global_ptr();
            let ret = (*ptr).as_ref().map(|s| (**s).clone());
            LOCK.unlock();
            return ret
        }
    }

    fn get_global_ptr() -> *mut Option<Box<Vec<Vec<u8>>>> {
        unsafe { mem::transmute(&GLOBAL_ARGS_PTR) }
    }

}
