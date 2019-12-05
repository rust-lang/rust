use crate::ffi::{CStr, OsStr, OsString};
use crate::marker::PhantomData;
use crate::os::wasi::ffi::OsStrExt;
use crate::vec;

pub unsafe fn init(_argc: isize, _argv: *const *const u8) {}

pub unsafe fn cleanup() {}

pub struct Args {
    iter: vec::IntoIter<OsString>,
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

/// Returns the command line arguments
pub fn args() -> Args {
    Args {
        iter: maybe_args().unwrap_or(Vec::new()).into_iter(),
        _dont_send_or_sync_me: PhantomData,
    }
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
