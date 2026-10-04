//@ edition:2021
//@ check-pass

#![feature(return_type_notation)]

use std::future::Future;

struct JoinHandle<T>(fn() -> T);

fn spawn<T>(_: impl Future<Output = T>) -> JoinHandle<T> {
    todo!()
}

trait Foo {
    async fn bar(&self) -> i32;
}

trait SendFoo: Foo<bar(..): Send> + Send {}

fn foobar(foo: impl SendFoo) -> JoinHandle<i32> {
    spawn(async move {
        let future = foo.bar();
        future.await
    })
}

fn main() {}
