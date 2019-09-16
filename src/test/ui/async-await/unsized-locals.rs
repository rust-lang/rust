// edition:2018

#![feature(unsized_locals)]
#![feature(gen_future)]

use std::future::poll_with_tls_context;
use std::pin::Pin;
use std::fmt::Display;

async fn foo2() {}

async fn foo(x: Box<dyn Display>) { //~ ERROR unsized values can't be used in `async` functions
    let x = *x;
    foo2().await;
    println!("hello {}", &x);
}

fn main() {
    let mut a = foo(Box::new(5));
    let b = unsafe {
        Pin::new_unchecked(&mut a)
    };
    match poll_with_tls_context(b) {
        _ => ()
    };
}
