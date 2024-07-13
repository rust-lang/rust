//@ check-pass
// https://github.com/rust-lang/rust/issues/100620

pub trait Bar<S> {}

pub trait Qux<T> {}

pub trait Foo<T, S> {
    fn bar()
    where
        T: Bar<S>,
    {
    }
}

pub struct Concrete;

impl<S> Foo<(), S> for Concrete {}

impl<T, S> Bar<S> for T where S: Qux<T> {}

impl<T, S> Qux<T> for S where T: Bar<S> {}
