// Regression test for #87549.
//@ incremental

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius
//@ [polonius] compile-flags: -Zpolonius=next

pub struct Table<T, const N: usize>([Option<T>; N]);

impl<'a, T, const N: usize> IntoIterator for &'a Table<T, N> {
    type IntoIter = std::iter::Flatten<std::slice::Iter<'a, T>>; //~ ERROR `&'a T` is not an iterator
    //~^ ERROR `&'a T` is not an iterator
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        //~^ ERROR `&'a T` is not an iterator
        //[nll]~| ERROR `&T` is not an iterator
        unimplemented!()
    }
}
fn main() {}
