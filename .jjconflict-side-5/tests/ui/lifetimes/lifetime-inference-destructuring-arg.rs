//! Regression test for https://github.com/rust-lang/rust/issues/13167

//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

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
