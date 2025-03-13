use crate::ffi::OsString;
use crate::fmt;

pub struct Args {
    i_forward: usize,
    i_back: usize,
    count: usize,
}

pub fn args() -> Args {
    panic!("args not implemented for succinct");
}

impl Args {
    #[cfg(target_os = "succinct-zkvm")]
    fn argv(_: usize) -> OsString {
        panic!("argv not implemented for succinct");
    }
}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().finish()
    }
}

impl Iterator for Args {
    type Item = OsString;

    fn next(&mut self) -> Option<OsString> {
        if self.i_forward >= self.count - self.i_back {
            None
        } else {
            let arg = Self::argv(self.i_forward);
            self.i_forward += 1;
            Some(arg)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize {
        self.count
    }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> {
        if self.i_back >= self.count - self.i_forward {
            None
        } else {
            let arg = Self::argv(self.count - 1 - self.i_back);
            self.i_back += 1;
            Some(arg)
        }
    }
}
