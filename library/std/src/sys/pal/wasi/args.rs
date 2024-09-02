#![forbid(unsafe_op_in_unsafe_fn)]

use crate::ffi::{CStr, OsStr, OsString};
use crate::os::wasi::ffi::OsStrExt;
use crate::{fmt, vec};

pub struct Args {
    iter: vec::IntoIter<OsString>,
}

impl !Send for Args {}
impl !Sync for Args {}

/// Returns the command line arguments
pub fn args() -> Args {
    Args { iter: maybe_args().unwrap_or(Vec::new()).into_iter() }
}

fn maybe_args() -> Option<Vec<OsString>> {
    unsafe {
        let (argc, buf_size) = wasi::args_sizes_get().ok()?;
        let mut argv = Vec::with_capacity(argc);
        let mut buf = Vec::with_capacity(buf_size);
        wasi::args_get(argv.as_mut_ptr(), buf.as_mut_ptr()).ok()?;
        argv.set_len(argc);
        let mut ret = Vec::with_capacity(argc);
        for ptr in argv {
            let s = CStr::from_ptr(ptr.cast());
            ret.push(OsStr::from_bytes(s.to_bytes()).to_owned());
        }
        Some(ret)
    }
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
