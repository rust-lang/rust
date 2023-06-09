// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// [drop_tracking] check-pass
// [drop_tracking_mir] check-pass

#![feature(negative_impls, generators)]

struct Foo;
impl !Send for Foo {}

fn main() {
    assert_send(|| {
        //[no_drop_tracking]~^ ERROR generator cannot be sent between threads safely
        let guard = Foo;
        drop(guard);
        yield;
    })
}

fn assert_send<T: Send>(_: T) {}
