//@ check-pass
//@ edition:2021

pub trait Foo {
    #[allow(async_fn_in_trait)]
    async fn foo(&mut self);
}

impl<T: Foo> Foo for &mut T {
    async fn foo(&mut self) {}
}

fn main() {}
