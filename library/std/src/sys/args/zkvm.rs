use crate::ffi::OsString;
use crate::fmt;
use crate::num::NonZero;
use crate::sys::os_str;
use crate::sys::pal::{WORD_SIZE, abi};
use crate::sys_common::FromInner;

#[derive(Clone)]
pub struct Args {
    front: usize,
    back: usize,
}

pub fn args() -> Args {
    let count = unsafe { abi::sys_argc() };
    Args { front: 0, back: count }
}

impl Args {
    /// Use sys_argv to get the arg at the requested index. Does not check that i is less than argc
    /// and will not return if the index is out of bounds.
    fn argv(i: usize) -> OsString {
        let arg_len = unsafe { abi::sys_argv(crate::ptr::null_mut(), 0, i) };

        let arg_len_words = (arg_len + WORD_SIZE - 1) / WORD_SIZE;
        let words = unsafe { abi::sys_alloc_words(arg_len_words) };

        let arg_len2 = unsafe { abi::sys_argv(words, arg_len_words, i) };
        debug_assert_eq!(arg_len, arg_len2);

        // Convert to OsString.
        //
        // FIXME: We can probably get rid of the extra copy here if we
        // reimplement "os_str" instead of just using the generic unix
        // "os_str".
        let arg_bytes: &[u8] =
            unsafe { crate::slice::from_raw_parts(words.cast() as *const u8, arg_len) };
        OsString::from_inner(os_str::Buf { inner: arg_bytes.to_vec() })
    }
}

impl !Send for Args {}
impl !Sync for Args {}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl Iterator for Args {
    type Item = OsString;

    #[inline]
    fn next(&mut self) -> Option<OsString> {
        if self.front == self.back {
            None
        } else {
            let arg = Self::argv(self.front);
            self.front += 1;
            Some(arg)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<OsString> {
        self.next_back()
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let step_size = self.len().min(n);
        self.front += step_size;
        NonZero::new(n - step_size).map_or(Ok(()), Err)
    }
}

impl DoubleEndedIterator for Args {
    #[inline]
    fn next_back(&mut self) -> Option<OsString> {
        if self.back == self.front {
            None
        } else {
            self.back -= 1;
            Some(Self::argv(self.back))
        }
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        let step_size = self.len().min(n);
        self.back -= step_size;
        NonZero::new(n - step_size).map_or(Ok(()), Err)
    }
}

impl ExactSizeIterator for Args {
    #[inline]
    fn len(&self) -> usize {
        self.back - self.front
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.front == self.back
    }
}
