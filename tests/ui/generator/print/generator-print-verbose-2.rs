// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// compile-flags: -Zverbose

// Same as test/ui/generator/not-send-sync.rs
#![feature(generators)]

use std::cell::Cell;

fn main() {
    fn assert_sync<T: Sync>(_: T) {}
    fn assert_send<T: Send>(_: T) {}

    assert_sync(|| {
        //~^ ERROR: generator cannot be shared between threads safely
        let a = Cell::new(2);
        yield;
    });

    let a = Cell::new(2);
    assert_send(|| {
        //~^ ERROR: E0277
        drop(&a);
        yield;
    });
}
