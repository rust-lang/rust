// edition:2021
#![feature(negative_impls)]
#![allow(unused)]

fn main() {
    gimme_send(foo());
    //~^ ERROR cannot be sent between threads safely
}

fn gimme_send<T: Send>(t: T) {
    drop(t);
}

struct NotSend {}

impl Drop for NotSend {
    fn drop(&mut self) {}
}

impl !Send for NotSend {}

async fn foo() {
    let mut x = (NotSend {},);
    drop(x.0);
    x.0 = NotSend {};
    bar().await;
}

async fn bar() {}
