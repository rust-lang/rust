use crate::ffi::OsString;
use crate::{fmt, vec};

pub struct Args {
    iter: vec::IntoIter<OsString>,
}

impl !Send for Args {}
impl !Sync for Args {}

impl Args {
    pub(super) fn new(args: Vec<OsString>) -> Self {
        Args { iter: args.into_iter() }
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
