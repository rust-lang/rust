//@ check-pass
//@ edition:2021

pub trait Foo {
    #[allow(async_fn_in_trait)]
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
