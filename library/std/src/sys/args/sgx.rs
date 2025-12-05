#![allow(fuzzy_provenance_casts)] // FIXME: this module systematically confuses pointers and integers

use crate::ffi::OsString;
use crate::num::NonZero;
use crate::ops::Try;
use crate::sync::atomic::{Atomic, AtomicUsize, Ordering};
use crate::sys::os_str::Buf;
use crate::sys::pal::abi::usercalls::alloc;
use crate::sys::pal::abi::usercalls::raw::ByteBuffer;
use crate::sys_common::FromInner;
use crate::{fmt, slice};

// Specifying linkage/symbol name is solely to ensure a single instance between this crate and its unit tests
#[cfg_attr(test, linkage = "available_externally")]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys3sgx4args4ARGSE")]
static ARGS: Atomic<usize> = AtomicUsize::new(0);
type ArgsStore = Vec<OsString>;

#[cfg_attr(test, allow(dead_code))]
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    if argc != 0 {
        let args = unsafe { alloc::User::<[ByteBuffer]>::from_raw_parts(argv as _, argc as _) };
        let args = args
            .iter()
            .map(|a| OsString::from_inner(Buf { inner: a.copy_user_buffer() }))
            .collect::<ArgsStore>();
        ARGS.store(Box::into_raw(Box::new(args)) as _, Ordering::Relaxed);
    }
}

pub fn args() -> Args {
    let args = unsafe { (ARGS.load(Ordering::Relaxed) as *const ArgsStore).as_ref() };
    let slice = args.map(|args| args.as_slice()).unwrap_or(&[]);
    Args { iter: slice.iter() }
}

pub struct Args {
    iter: slice::Iter<'static, OsString>,
}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.iter.as_slice().fmt(f)
    }
}

impl Iterator for Args {
    type Item = OsString;

    fn next(&mut self) -> Option<OsString> {
        self.iter.next().cloned()
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
        self.iter.last().cloned()
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.iter.advance_by(n)
    }

    fn try_fold<B, F, R>(&mut self, init: B, f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        self.iter.by_ref().cloned().try_fold(init, f)
    }

    fn fold<B, F>(self, init: B, f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        self.iter.cloned().fold(init, f)
    }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> {
        self.iter.next_back().cloned()
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
