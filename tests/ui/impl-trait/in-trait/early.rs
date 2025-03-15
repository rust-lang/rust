//@ check-pass
//@ edition:2021

pub trait Foo {
    async fn bar<'a: 'a>(&'a mut self);
}

impl Foo for () {
    async fn bar<'a: 'a>(&'a mut self) {}
}

pub trait Foo2 {
    fn bar<'a: 'a>(&'a mut self) -> impl Sized + 'a;
}

impl Foo2 for () {
    fn bar<'a: 'a>(&'a mut self) -> impl Sized + 'a {}
}

fn main() {}
