use crate::ffi::{c_char, CStr, OsString};
use crate::fmt;
use crate::os::hermit::ffi::OsStringExt;
use crate::ptr;
use crate::sync::atomic::{
    AtomicIsize, AtomicPtr,
    Ordering::{Acquire, Relaxed, Release},
};
use crate::vec;

static ARGC: AtomicIsize = AtomicIsize::new(0);
static ARGV: AtomicPtr<*const u8> = AtomicPtr::new(ptr::null_mut());

/// One-time global initialization.
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    ARGC.store(argc, Relaxed);
    // Use release ordering here to broadcast writes by the OS.
    ARGV.store(argv as *mut *const u8, Release);
}

/// Returns the command line arguments
pub fn args() -> Args {
    // Synchronize with the store above.
    let argv = ARGV.load(Acquire);
    // If argv has not been initialized yet, do not return any arguments.
    let argc = if argv.is_null() { 0 } else { ARGC.load(Relaxed) };
    let args: Vec<OsString> = (0..argc)
        .map(|i| unsafe {
            let cstr = CStr::from_ptr(*argv.offset(i) as *const c_char);
            OsStringExt::from_vec(cstr.to_bytes().to_vec())
        })
        .collect();

    Args { iter: args.into_iter() }
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
