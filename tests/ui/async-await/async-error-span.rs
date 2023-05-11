// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// edition:2018

// Regression test for issue #62382.

use std::future::Future;

fn get_future() -> impl Future<Output = ()> {
//~^ ERROR `()` is not a future
    panic!()
}

async fn foo() {
    let a;
    //[no_drop_tracking,drop_tracking]~^ ERROR type inside `async fn` body must be known in this context
    //[drop_tracking_mir]~^^ ERROR type annotations needed
    get_future().await;
}

fn main() {}
