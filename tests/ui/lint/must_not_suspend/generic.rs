//@ edition:2018
//@ run-pass
//
// this test shows a case where the lint doesn't fire in generic code
#![feature(must_not_suspend)]
#![deny(must_not_suspend)]

#[must_not_suspend]
struct No {}

async fn shushspend() {}

async fn wheeee<T>(t: T) {
    shushspend().await;
    drop(t);
}

fn main() {
    let _fut = wheeee(No {});
}
