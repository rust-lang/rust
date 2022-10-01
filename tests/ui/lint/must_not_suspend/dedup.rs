// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// edition:2018
#![feature(must_not_suspend)]
#![deny(must_not_suspend)]

#[must_not_suspend]
struct No {}

async fn shushspend() {}

async fn wheeee<T>(t: T) {
    shushspend().await;
    drop(t);
}

async fn yes() {
    wheeee(&No {}).await; //~ ERROR `No` held across
}

fn main() {
}
