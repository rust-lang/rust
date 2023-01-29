// check-pass
// compile-flags: -Zdrop-tracking

#![feature(negative_impls, generators)]

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
