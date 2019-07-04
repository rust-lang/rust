// build-pass (FIXME(62277): could be check-pass?)
// Regression test for #37655. The problem was a false edge created by
// coercion that wound up requiring that `'a` (in `split()`) outlive
// `'b`, which shouldn't be necessary.

#![allow(warnings)]

trait SliceExt<T> {
    type Item;

    fn get_me<I>(&self, index: I) -> &I::Output
        where I: SliceIndex<Self::Item>;
}

impl<T> SliceExt<T> for [T] {
    type Item = T;

    fn get_me<I>(&self, index: I) -> &I::Output
        where I: SliceIndex<T>
    {
        panic!()
    }
}

pub trait SliceIndex<T> {
    type Output: ?Sized;
}

impl<T> SliceIndex<T> for usize {
    type Output = T;
}

fn foo<'a, 'b>(split: &'b [&'a [u8]]) -> &'a [u8] {
    split.get_me(0)
}

fn main() { }
