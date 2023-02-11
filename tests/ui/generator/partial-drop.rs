// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// [drop_tracking_mir] check-pass

#![feature(negative_impls, generators)]

struct Foo;
impl !Send for Foo {}

struct Bar {
    foo: Foo,
    x: i32,
}

fn main() {
    assert_send(|| {
        //[no_drop_tracking,drop_tracking]~^ ERROR generator cannot be sent between threads safely
        let guard = Bar { foo: Foo, x: 42 };
        drop(guard.foo);
        yield;
    });

    assert_send(|| {
        //[no_drop_tracking,drop_tracking]~^ ERROR generator cannot be sent between threads safely
        let guard = Bar { foo: Foo, x: 42 };
        let Bar { foo, x } = guard;
        drop(foo);
        yield;
    });
}

fn assert_send<T: Send>(_: T) {}
