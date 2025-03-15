//@ run-rustfix
//@ edition: 2021

#![allow(unused)]

trait Foo {
    async fn test() -> () {}
    async fn test2() -> i32 { 1 + 2 }
}

fn needs_send(_: impl Send) {}

fn bar<T: Foo>() {
    needs_send(T::test());
    //~^ ERROR `impl Future<Output = ()> { <T as Foo>::test(..) }` cannot be sent between threads safely
}

fn bar2<T: Foo>() {
    needs_send(T::test2());
    //~^ ERROR `impl Future<Output = i32> { <T as Foo>::test2(..) }` cannot be sent between threads safely
}

fn main() {}
