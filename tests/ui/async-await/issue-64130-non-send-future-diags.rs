// edition:2018
#![feature(must_not_suspend)]
#![allow(must_not_suspend)]

// This tests the basic example case for the async-await-specific error.

use std::sync::Mutex;

fn is_send<T: Send>(t: T) { }

async fn foo() {
    bar(&Mutex::new(22)).await;
}

async fn bar(x: &Mutex<u32>) {
    let g = x.lock().unwrap();
    baz().await;
}

async fn baz() { }

fn main() {
    is_send(foo());
    //~^ ERROR future cannot be sent between threads safely
}
