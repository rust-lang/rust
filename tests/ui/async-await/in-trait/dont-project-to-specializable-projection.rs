//@ ignore-backends: gcc
//@ edition: 2021
//@ known-bug: #108309

#![feature(min_specialization)]

struct MyStruct;

trait MyTrait<T> {
    async fn foo(_: T) -> &'static str;
}

impl<T> MyTrait<T> for MyStruct {
    default async fn foo(_: T) -> &'static str {
        "default"
    }
}

impl MyTrait<i32> for MyStruct {
    async fn foo(_: i32) -> &'static str {
        "specialized"
    }
}

async fn async_main() {
    assert_eq!(MyStruct::foo(42).await, "specialized");
    assert_eq!(indirection(42).await, "specialized");
}

async fn indirection<T>(x: T) -> &'static str {
    //explicit type coercion is currently necessary
    // because of https://github.com/rust-lang/rust/issues/67918
    <MyStruct as MyTrait<T>>::foo(x).await
}

// ------------------------------------------------------------------------- //
// Implementation Details Below...

use std::pin::{pin, Pin};
use std::task::*;

fn main() {
    let mut fut = pin!(async_main());

    // Poll loop, just to test the future...
    let ctx = &mut Context::from_waker(Waker::noop());

    loop {
        match fut.as_mut().poll(ctx) {
            Poll::Pending => {}
            Poll::Ready(()) => break,
        }
    }
}
