use crate::convert::AsRef;
use crate::ffi::OsString;
use crate::fmt;
use crate::sys::args as sys;

pub struct Args(sys::Args);

impl !Send for Args {}
impl !Sync for Args {}

impl Args {
    pub fn get() -> Args {
        Args(sys::args())
    }
}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let args = &AsRef::<[OsString]>::as_ref(self);
        args.fmt(f)
    }
}

impl AsRef<[OsString]> for Args {
    fn as_ref(&self) -> &[OsString] {
        self.0.as_ref()
    }
}

impl Iterator for Args {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> {
        self.0.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> {
        self.0.next_back()
    }
}
