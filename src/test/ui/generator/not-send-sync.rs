#![feature(generators)]

use std::cell::Cell;

fn main() {
    fn assert_sync<T: Sync>(_: T) {}
    fn assert_send<T: Send>(_: T) {}

    assert_sync(|| {
        //~^ ERROR: future cannot be shared between threads safely
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
