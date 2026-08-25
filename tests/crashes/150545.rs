//@ known-bug: #150545
#![feature(non_lifetime_binders)]
trait Foo: for<T> Bar<T> {
    type Item;
    fn next(self) -> Self::Item;
}
trait Bar<T> {}
fn main() {}
