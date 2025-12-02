//@ edition:2021
#![allow(todo_macro_calls)]


trait Foo {
    async fn foo(&self);
}

fn main() {
    let x: &dyn Foo = todo!();
    //~^ ERROR the trait `Foo` is not dyn compatible
}
