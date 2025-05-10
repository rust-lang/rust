use crate::ffi::{OsStr, OsString};
use crate::num::NonZero;
use crate::sync::OnceLock;
use crate::sys::pal::{WORD_SIZE, abi};
use crate::{fmt, ptr, slice};

pub fn args() -> Args {
    Args { iter: ARGS.get_or_init(|| get_args()).iter() }
}

fn get_args() -> Vec<&'static OsStr> {
    let argc = unsafe { abi::sys_argc() };
    let mut args = Vec::with_capacity(argc);

    for i in 0..argc {
        // Get the size of the argument then the data.
        let arg_len = unsafe { abi::sys_argv(ptr::null_mut(), 0, i) };

        let arg_len_words = (arg_len + WORD_SIZE - 1) / WORD_SIZE;
        let words = unsafe { abi::sys_alloc_words(arg_len_words) };

        let arg_len2 = unsafe { abi::sys_argv(words, arg_len_words, i) };
        debug_assert_eq!(arg_len, arg_len2);

        let arg_bytes = unsafe { slice::from_raw_parts(words.cast(), arg_len) };
        args.push(unsafe { OsStr::from_encoded_bytes_unchecked(arg_bytes) });
    }
    args
}

static ARGS: OnceLock<Vec<&'static OsStr>> = OnceLock::new();

pub struct Args {
    iter: slice::Iter<'static, &'static OsStr>,
}

impl !Send for Args {}
impl !Sync for Args {}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.iter.as_slice().fmt(f)
    }
}

impl Iterator for Args {
    type Item = OsString;

    fn next(&mut self) -> Option<OsString> {
        self.iter.next().map(|arg| arg.to_os_string())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.len()
    }

    fn last(self) -> Option<OsString> {
        self.iter.last().map(|arg| arg.to_os_string())
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.iter.advance_by(n)
    }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> {
        self.iter.next_back().map(|arg| arg.to_os_string())
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.iter.advance_back_by(n)
    }
}

impl ExactSizeIterator for Args {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}
