// Regression test for #60218
//
// This was reported to cause ICEs.

use std::iter::Map;

pub trait Foo {}

pub fn trigger_error<I, F>(iterable: I, functor: F)
where
    for<'t> &'t I: IntoIterator,
for<'t> Map<<&'t I as IntoIterator>::IntoIter, F>: Iterator,
for<'t> <Map<<&'t I as IntoIterator>::IntoIter, F> as Iterator>::Item: Foo,
{
}

fn main() {
    trigger_error(vec![], |x: &u32| x) //~ ERROR E0277
}
