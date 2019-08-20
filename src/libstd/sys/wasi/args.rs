use crate::ffi::CStr;
use crate::io;
use crate::ffi::OsString;
use crate::marker::PhantomData;
use crate::os::wasi::ffi::OsStringExt;
use crate::vec;

pub unsafe fn init(_argc: isize, _argv: *const *const u8) {
}

pub unsafe fn cleanup() {
}

pub struct Args {
    iter: vec::IntoIter<OsString>,
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

/// Returns the command line arguments
pub fn args() -> Args {
    let mut buf = Vec::new();
    let _ = wasi::get_args(|arg| buf.push(OsString::from_vec(arg.to_vec())));
    Args {
        iter: buf.into_iter(),
        _dont_send_or_sync_me: PhantomData
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
