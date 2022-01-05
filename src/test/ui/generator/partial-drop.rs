#![feature(negative_impls, generators)]

struct Foo;
impl !Send for Foo {}

struct Bar {
    foo: Foo,
    x: i32,
}

fn main() {
    assert_send(|| {
        //~^ ERROR generator cannot be sent between threads safely
        // FIXME: it would be nice to make this work.
        let guard = Bar { foo: Foo, x: 42 };
        drop(guard.foo);
        yield;
    })
}

fn assert_send<T: Send>(_: T) {}
