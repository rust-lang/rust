// aux-build:foreign-async-fn.rs
// edition:2021

#![feature(async_fn_in_trait)]

extern crate foreign_async_fn;
use foreign_async_fn::Foo;

fn bar<T: Foo>() {
    fn needs_send(_: impl Send) {}
    needs_send(T::test());
    //~^ ERROR `impl Future<Output = ()>` cannot be sent between threads safely
}

fn main() {}
