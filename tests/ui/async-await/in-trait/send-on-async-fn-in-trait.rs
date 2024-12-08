//@ run-rustfix
//@ edition: 2021

#![allow(unused)]

trait Foo {
    async fn test() -> () {}
    async fn test2() -> i32 { 1 + 2 }
}

fn bar<T: Foo>() {
    fn needs_send(_: impl Send) {}
    needs_send(T::test());
    //~^ ERROR `impl Future<Output = ()>` cannot be sent between threads safely
    needs_send(T::test2());
    //~^ ERROR `impl Future<Output = i32>` cannot be sent between threads safely
}

fn main() {}
