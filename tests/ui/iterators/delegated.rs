//@ check-pass

// This is a regression test for:
// - https://github.com/rust-lang/rust/issues/109632
// - https://github.com/rust-lang/rust/issues/166229
// (It's the same regression, we just didn't add a regression test. Now we have.)

use std::marker::PhantomData;
use std::ops::Deref;

pub trait Indirect {
    type Output;
}

pub struct Wrapper<I: Indirect> {
    inner: PhantomData<I>,
}

impl<I: Indirect> Deref for Wrapper<I> {
    type Target = I::Output;
    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}

impl<I: Indirect> std::ops::DerefMut for Wrapper<I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unimplemented!()
    }
}

// NOTE: the regression here is that the signatures of the below don't actually match the bounds
// on these methods, since the trait solver wasn't able to determine that blanket impls on
// `Destruct` trivially apply to the given type for non-const implementations. which methods we
// choose to delegate here is mostly arbitrary, but we use manual delegation since the original
// regression was for automatically generated code.

impl<I: Indirect> Iterator for Wrapper<I>
where
    <Self as Deref>::Target: Iterator,
{
    type Item = <<Self as Deref>::Target as Iterator>::Item;
    fn next(&mut self) -> Option<<<Self as Deref>::Target as Iterator>::Item> {
        (**self).next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
    fn nth(&mut self, n: usize) -> Option<<<Self as Deref>::Target as Iterator>::Item> {
        (**self).nth(n)
    }
}

impl<I: Indirect> DoubleEndedIterator for Wrapper<I>
where
    <Self as Deref>::Target: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<<<Self as Deref>::Target as Iterator>::Item> {
        (**self).next_back()
    }
    fn nth_back(&mut self, n: usize) -> Option<<<Self as Deref>::Target as Iterator>::Item> {
        (**self).nth_back(n)
    }
}

fn main() {}
