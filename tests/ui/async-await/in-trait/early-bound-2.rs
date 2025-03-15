//@ check-pass
//@ edition:2021

pub trait Foo {
    async fn foo(&mut self);
}

impl<T: Foo> Foo for &mut T {
    async fn foo(&mut self) {}
}

fn main() {}
