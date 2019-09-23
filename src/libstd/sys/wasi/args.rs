use crate::ffi::OsString;
use crate::marker::PhantomData;
use crate::os::wasi::ffi::OsStringExt;
use crate::vec;

use ::wasi::wasi_unstable as wasi;

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
    let buf = wasi::args_sizes_get().and_then(|args_sizes| {
        let mut buf = Vec::with_capacity(args_sizes.get_count());
        wasi::args_get(args_sizes, |arg| {
            let arg = OsString::from_vec(arg.to_vec());
            buf.push(arg);
        })?;
        Ok(buf)
    }).unwrap_or(vec![]);
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
