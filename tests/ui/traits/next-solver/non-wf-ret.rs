//@ check-pass
//@ compile-flags: -Znext-solver

use std::ops::Deref;

pub struct List<T> {
    skel: [T],
}

impl<'a, T: Copy> IntoIterator for &'a List<T> {
    type Item = T;
    type IntoIter = std::iter::Copied<<&'a [T] as IntoIterator>::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

impl<T> Deref for List<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        todo!()
    }
}

impl<T> List<T> {
    fn iter(&self) -> <&Self as IntoIterator>::IntoIter
    where
        T: Copy,
    {
        todo!()
    }
}

fn test<Q>(t: &List<Q>) {
    // Checking that `<&List<Q> as IntoIterator>::IntoIter` is WF
    // will disqualify the inherent method, since normalizing it
    // requires `Q: Copy` which does not hold. and allow us to fall
    // through to the deref'd `<[Q]>::iter` method which works.
    //
    // In the old solver, the same behavior is achieved by just
    // eagerly normalizing the return type.
    t.iter();
}

fn main() {}
