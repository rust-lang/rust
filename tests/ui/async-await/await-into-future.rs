//@ run-pass
//@ aux-build: issue-72470-lib.rs
//@ edition:2021
extern crate issue_72470_lib;
use std::{future::{Future, IntoFuture}, pin::Pin};

struct AwaitMe;

impl IntoFuture for AwaitMe {
    type Output = i32;
    type IntoFuture = Pin<Box<dyn Future<Output = i32>>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(me())
    }
}

async fn me() -> i32 {
    41
}

async fn run() {
    assert_eq!(AwaitMe.await, 41);
}

fn main() {
    issue_72470_lib::run(run());
}
