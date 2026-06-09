use crate::ffi::OsString;
use crate::{fmt, vec};

pub struct Env {
    iter: vec::IntoIter<(OsString, OsString)>,
}

impl Env {
    pub(super) fn new(env: Vec<(OsString, OsString)>) -> Self {
        Env { iter: env.into_iter() }
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter.as_slice()).finish()
    }
}

impl !Send for Env {}
impl !Sync for Env {}

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
