// check-pass

// FIXME(eholk): temporarily disabled while drop range tracking is disabled
// (see generator_interior.rs:27)
// ignore-test

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
