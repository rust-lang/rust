//! Global initialization and retrieval of command line arguments.
//!
//! On some platforms these are stored during runtime startup,
//! and on some they are retrieved from the system on demand.

#![allow(dead_code)] // runtime init functions not used during testing

use crate::ffi::OsString;
use crate::marker::PhantomData;
use crate::vec;

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

impl Args {
    pub fn inner_debug(&self) -> &[OsString] {
        self.iter.as_slice()
    }
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
    use crate::os::unix::prelude::*;
    use crate::mem;
    use crate::ffi::{CStr, OsString};
    use crate::marker::PhantomData;
    use super::Args;

    use crate::sys_common::mutex::Mutex;

    static mut GLOBAL_ARGS_PTR: usize = 0;
    static LOCK: Mutex = Mutex::new();

    pub unsafe fn init(argc: isize, argv: *const *const u8) {
        let args = (0..argc).map(|i| {
            CStr::from_ptr(*argv.offset(i) as *const libc::c_char).to_bytes().to_vec()
        }).collect();

        let _guard = LOCK.lock();
        let ptr = get_global_ptr();
        assert!((*ptr).is_none());
        (*ptr) = Some(box args);
    }

    pub unsafe fn cleanup() {
        let _guard = LOCK.lock();
        *get_global_ptr() = None;
    }

    pub fn args() -> Args {
        let bytes = clone().unwrap_or_default();
        let v: Vec<OsString> = bytes.into_iter().map(|v| {
            OsStringExt::from_vec(v)
        }).collect();
        Args { iter: v.into_iter(), _dont_send_or_sync_me: PhantomData }
    }

    fn clone() -> Option<Vec<Vec<u8>>> {
        unsafe {
            let _guard = LOCK.lock();
            let ptr = get_global_ptr();
            (*ptr).as_ref().map(|s| (**s).clone())
        }
    }

    unsafe fn get_global_ptr() -> *mut Option<Box<Vec<Vec<u8>>>> {
        mem::transmute(&GLOBAL_ARGS_PTR)
    }

}
