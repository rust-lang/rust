//@ check-pass
//@ edition:2021

pub trait Foo {
    #[allow(async_fn_in_trait)]
    async fn foo(&mut self);
}

struct MyFoo<'a>(&'a mut ());

impl<'a> Foo for MyFoo<'a> {
    async fn foo(&mut self) {}
}

fn main() {}
