// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
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

#[cfg(any(target_os = "linux",
          target_os = "android",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "netbsd",
          target_os = "openbsd",
          target_os = "solaris",
          target_os = "emscripten",
          target_os = "haiku",
          target_os = "fuchsia"))]
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

#[cfg(any(target_os = "macos",
          target_os = "ios"))]
mod imp {
    use ffi::CStr;
    use marker::PhantomData;
    use libc;
    use super::Args;

    pub unsafe fn init(_argc: isize, _argv: *const *const u8) {
    }

    pub fn cleanup() {
    }

    #[cfg(target_os = "macos")]
    pub fn args() -> Args {
        use os::unix::prelude::*;
        extern {
            // These functions are in crt_externs.h.
            fn _NSGetArgc() -> *mut libc::c_int;
            fn _NSGetArgv() -> *mut *mut *mut libc::c_char;
        }

        let vec = unsafe {
            let (argc, argv) = (*_NSGetArgc() as isize,
                                *_NSGetArgv() as *const *const libc::c_char);
            (0.. argc as isize).map(|i| {
                let bytes = CStr::from_ptr(*argv.offset(i)).to_bytes().to_vec();
                OsStringExt::from_vec(bytes)
            }).collect::<Vec<_>>()
        };
        Args {
            iter: vec.into_iter(),
            _dont_send_or_sync_me: PhantomData,
        }
    }

    // As _NSGetArgc and _NSGetArgv aren't mentioned in iOS docs
    // and use underscores in their names - they're most probably
    // are considered private and therefore should be avoided
    // Here is another way to get arguments using Objective C
    // runtime
    //
    // In general it looks like:
    // res = Vec::new()
    // let args = [[NSProcessInfo processInfo] arguments]
    // for i in (0..[args count])
    //      res.push([args objectAtIndex:i])
    // res
    #[cfg(target_os = "ios")]
    pub fn args() -> Args {
        use ffi::OsString;
        use mem;
        use str;

        extern {
            fn sel_registerName(name: *const libc::c_uchar) -> Sel;
            fn objc_getClass(class_name: *const libc::c_uchar) -> NsId;
        }

        #[cfg(target_arch="aarch64")]
        extern {
            fn objc_msgSend(obj: NsId, sel: Sel) -> NsId;
            #[link_name="objc_msgSend"]
            fn objc_msgSend_ul(obj: NsId, sel: Sel, i: libc::c_ulong) -> NsId;
        }

        #[cfg(not(target_arch="aarch64"))]
        extern {
            fn objc_msgSend(obj: NsId, sel: Sel, ...) -> NsId;
            #[link_name="objc_msgSend"]
            fn objc_msgSend_ul(obj: NsId, sel: Sel, ...) -> NsId;
        }

        #[link(name = "Foundation", kind = "framework")]
        #[link(name = "objc")]
        #[cfg(not(cargobuild))]
        extern {}

        type Sel = *const libc::c_void;
        type NsId = *const libc::c_void;

        let mut res = Vec::new();

        unsafe {
            let process_info_sel = sel_registerName("processInfo\0".as_ptr());
            let arguments_sel = sel_registerName("arguments\0".as_ptr());
            let utf8_sel = sel_registerName("UTF8String\0".as_ptr());
            let count_sel = sel_registerName("count\0".as_ptr());
            let object_at_sel = sel_registerName("objectAtIndex:\0".as_ptr());

            let klass = objc_getClass("NSProcessInfo\0".as_ptr());
            let info = objc_msgSend(klass, process_info_sel);
            let args = objc_msgSend(info, arguments_sel);

            let cnt: usize = mem::transmute(objc_msgSend(args, count_sel));
            for i in 0..cnt {
                let tmp = objc_msgSend_ul(args, object_at_sel, i as libc::c_ulong);
                let utf_c_str: *const libc::c_char =
                    mem::transmute(objc_msgSend(tmp, utf8_sel));
                let bytes = CStr::from_ptr(utf_c_str).to_bytes();
                res.push(OsString::from(str::from_utf8(bytes).unwrap()))
            }
        }

        Args { iter: res.into_iter(), _dont_send_or_sync_me: PhantomData }
    }
}
