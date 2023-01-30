// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir

// edition:2018
// Test that impl trait does not allow creating recursive types that are
// otherwise forbidden when using `async` and `await`.

async fn rec_1() { //~ ERROR recursion in an `async fn`
    rec_2().await;
}

async fn rec_2() { //~ ERROR recursion in an `async fn`
    rec_1().await;
}

fn main() {}
