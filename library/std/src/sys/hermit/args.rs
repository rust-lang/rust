use crate::ffi::OsString;
use crate::fmt;
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

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.iter.as_slice().fmt(f)
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
    use crate::sys_common::os_str_bytes::*;

    use crate::sys::mutex::Mutex;

    static mut ARGC: isize = 0;
    static mut ARGV: *const *const u8 = ptr::null();
    static LOCK: Mutex = Mutex::new();

    pub unsafe fn init(argc: isize, argv: *const *const u8) {
        LOCK.lock();
        ARGC = argc;
        ARGV = argv;
        LOCK.unlock();
    }

    pub unsafe fn cleanup() {
        LOCK.lock();
        ARGC = 0;
        ARGV = ptr::null();
        LOCK.unlock();
    }

    pub fn args() -> Args {
        Args { iter: clone().into_iter(), _dont_send_or_sync_me: PhantomData }
    }

    fn clone() -> Vec<OsString> {
        unsafe {
            LOCK.lock();
            let argc = ARGC;
            let argv = ARGV;
            LOCK.unlock();

            (0..argc)
                .map(|i| {
                    let cstr = CStr::from_ptr(*argv.offset(i) as *const i8);
                    OsStringExt::from_vec(cstr.to_bytes().to_vec())
                })
                .collect()
        }
    }
}
