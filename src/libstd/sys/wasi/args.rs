use crate::any::Any;
use crate::ffi::CStr;
use crate::ffi::OsString;
use crate::marker::PhantomData;
use crate::os::wasi::ffi::OsStringExt;
use crate::ptr;
use crate::vec;

static mut ARGC: isize = 0;
static mut ARGV: *const *const u8 = ptr::null();

#[cfg(not(target_feature = "atomics"))]
pub unsafe fn args_lock() -> impl Any {
    // No need for a lock if we're single-threaded, but this function will need
    // to get implemented for multi-threaded scenarios
}

pub unsafe fn init(argc: isize, argv: *const *const u8) {
    let _guard = args_lock();
    ARGC = argc;
    ARGV = argv;
}

pub unsafe fn cleanup() {
    let _guard = args_lock();
    ARGC = 0;
    ARGV = ptr::null();
}

pub struct Args {
    iter: vec::IntoIter<OsString>,
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

/// Returns the command line arguments
pub fn args() -> Args {
    unsafe {
        let _guard = args_lock();
        let args = (0..ARGC)
            .map(|i| {
                let cstr = CStr::from_ptr(*ARGV.offset(i) as *const libc::c_char);
                OsStringExt::from_vec(cstr.to_bytes().to_vec())
            })
            .collect::<Vec<_>>();
        Args {
            iter: args.into_iter(),
            _dont_send_or_sync_me: PhantomData,
        }
    }
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
