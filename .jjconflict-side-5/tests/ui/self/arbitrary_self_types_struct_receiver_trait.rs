//@ run-pass
#![feature(arbitrary_self_types)]

use std::ops::Receiver;

struct SmartPtr<T>(T);

impl<T> Receiver for SmartPtr<T> {
    type Target = T;
}

struct Foo {
    x: i32,
    y: i32,
}

impl Foo {
    fn x(self: &SmartPtr<Self>) -> i32 {
        self.0.x
    }

    fn y(self: SmartPtr<Self>) -> i32 {
        self.0.y
    }
}

fn main() {
    let foo = SmartPtr(Foo {x: 3, y: 4});
    assert_eq!(3, foo.x());
    assert_eq!(4, foo.y());
}
