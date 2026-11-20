use crate::ffi::OsString;
use crate::fmt;

pub struct Args {}

pub fn args() -> Args {
    Args {}
}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().finish()
    }
}

impl Iterator for Args {
    type Item = OsString;

    #[inline]
    fn next(&mut self) -> Option<OsString> {
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

impl DoubleEndedIterator for Args {
    #[inline]
    fn next_back(&mut self) -> Option<OsString> {
        None
    }
}

impl ExactSizeIterator for Args {
    #[inline]
    fn len(&self) -> usize {
        0
    }
}
