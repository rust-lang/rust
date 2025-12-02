//@ aux-build: rpitit.rs
#![allow(todo_macro_calls)]

extern crate rpitit;

fn main() {
    let _: &dyn rpitit::Foo = todo!();
    //~^ ERROR the trait `Foo` is not dyn compatible
}
