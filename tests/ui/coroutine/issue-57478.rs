//@ check-pass

#![feature(negative_impls, coroutines)]

struct Foo;
impl !Send for Foo {}

fn main() {
    assert_send(|| {
        let guard = Foo;
        drop(guard);
        yield;
    })
}

fn assert_send<T: Send>(_: T) {}
