#![allow(dead_code)] // runtime init functions not used during testing
use crate::ffi::OsString;
use crate::marker::PhantomData;
use crate::vec;

/// One-time global initialization.
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    imp::init(argc, argv)
}

/// One-time global cleanup.
pub unsafe fn cleanup() {
    imp::cleanup()
}

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
    fn next(&mut self) -> Option<OsString> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> {
        self.iter.next_back()
    }
}

mod imp {
    use super::Args;
    use crate::ffi::{CStr, OsString};
    use crate::marker::PhantomData;
    use crate::ptr;

    use crate::sys_common::mutex::StaticMutex;

    static mut ARGC: isize = 0;
    static mut ARGV: *const *const u8 = ptr::null();
    static LOCK: StaticMutex = StaticMutex::new();

    pub unsafe fn init(argc: isize, argv: *const *const u8) {
        let _guard = LOCK.lock();
        ARGC = argc;
        ARGV = argv;
    }

    pub unsafe fn cleanup() {
        let _guard = LOCK.lock();
        ARGC = 0;
        ARGV = ptr::null();
    }

    pub fn args() -> Args {
        Args { iter: clone().into_iter(), _dont_send_or_sync_me: PhantomData }
    }

    fn clone() -> Vec<OsString> {
        unsafe {
            let _guard = LOCK.lock();
            let ret = (0..ARGC)
                .map(|i| {
                    let cstr = CStr::from_ptr(*ARGV.offset(i) as *const libc::c_char);
                    use crate::sys::vxworks::ext::ffi::OsStringExt;
                    OsStringExt::from_vec(cstr.to_bytes().to_vec())
                })
                .collect();
            return ret;
        }
    }
}
