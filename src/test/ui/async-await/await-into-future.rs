// check-pass

// edition:2018

#![feature(into_future)]

use std::{future::{Future, IntoFuture}, pin::Pin};

struct AwaitMe;

impl IntoFuture for AwaitMe {
    type Output = i32;
    type Future = Pin<Box<dyn Future<Output = i32>>>;

    fn into_future(self) -> Self::Future {
        Box::pin(me())
    }
}

async fn me() -> i32 {
    41
}

async fn run() {
    assert_eq!(AwaitMe.await, 41);
}

fn main() {}
