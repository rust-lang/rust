//@ run-pass
#![allow(unused_variables)]
// Test that we can resolve nested projection types. Issue #20666.


use std::slice;

trait Bound {}

impl<'a> Bound for &'a i32 {}

trait IntoIterator {
    type Iter: Iterator;

    fn into_iter(self) -> Self::Iter; //~ WARN method `into_iter` is never used
}

impl<'a, T> IntoIterator for &'a [T; 3] {
    type Iter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

fn foo<X>(x: X) where
    X: IntoIterator,
    <<X as IntoIterator>::Iter as Iterator>::Item: Bound,
{
}

fn bar<T, I, X>(x: X) where
    T: Bound,
    I: Iterator<Item=T>,
    X: IntoIterator<Iter=I>,
{

}

fn main() {
    foo(&[0, 1, 2]);
    bar(&[0, 1, 2]);
}
