//@ known-bug: rust-lang/rust#129850

pub trait Foo2 {
    fn bar<'a: 'a>(&'a mut self) -> impl Sized + use<'static>;
}

impl Foo2 for () {
    fn bar<'a: 'a>(&'a mut self) -> impl Sized + 'a {}
}
