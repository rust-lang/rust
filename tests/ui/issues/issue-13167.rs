// check-pass
// pretty-expanded FIXME #23616
// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

use std::slice;

pub struct PhfMapEntries<'a, T: 'a> {
    iter: slice::Iter<'a, (&'static str, T)>,
}

impl<'a, T> Iterator for PhfMapEntries<'a, T> {
    type Item = (&'static str, &'a T);

    fn next(&mut self) -> Option<(&'static str, &'a T)> {
        self.iter.by_ref().map(|&(key, ref value)| (key, value)).next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

fn main() {}
