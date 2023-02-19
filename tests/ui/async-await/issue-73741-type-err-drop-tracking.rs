// edition:2018
// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
//
// Regression test for issue #73741
// Ensures that we don't emit spurious errors when
// a type error ocurrs in an `async fn`

async fn weird() {
    1 = 2; //~ ERROR invalid left-hand side

    let mut loop_count = 0;
    async {}.await
}

fn main() {}
