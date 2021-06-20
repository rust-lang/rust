use crate::ffi::OsString;
use crate::fmt;
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
}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.iter.as_slice().fmt(f)
    }
}

impl !Send for Args {}
impl !Sync for Args {}

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
    use crate::os::unix::ffi::OsStringExt;
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
        Args { iter: clone().into_iter() }
    }

    fn clone() -> Vec<OsString> {
        unsafe {
            let _guard = LOCK.lock();
            (0..ARGC)
                .map(|i| {
                    let cstr = CStr::from_ptr(*ARGV.offset(i) as *const i8);
                    OsStringExt::from_vec(cstr.to_bytes().to_vec())
                })
                .collect()
        }
    }
}
