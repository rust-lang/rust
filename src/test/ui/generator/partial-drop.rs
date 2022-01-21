// FIXME(eholk): temporarily disabled while drop range tracking is disabled
// (see generator_interior.rs:27)
// ignore-test

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
    });

    assert_send(|| {
        //~^ ERROR generator cannot be sent between threads safely
        // FIXME: it would be nice to make this work.
        let guard = Bar { foo: Foo, x: 42 };
        drop(guard);
        guard.foo = Foo;
        guard.x = 23;
        yield;
    });

    assert_send(|| {
        //~^ ERROR generator cannot be sent between threads safely
        // FIXME: it would be nice to make this work.
        let guard = Bar { foo: Foo, x: 42 };
        let Bar { foo, x } = guard;
        drop(foo);
        yield;
    });
}

fn assert_send<T: Send>(_: T) {}
