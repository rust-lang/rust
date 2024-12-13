//@ known-bug: #133868

trait Foo {
    type Assoc;
}

trait Bar {
    fn method() -> impl Sized;
}
impl<T> Bar for T where <T as Foo>::Assoc: Sized
{
    fn method() {}
}
