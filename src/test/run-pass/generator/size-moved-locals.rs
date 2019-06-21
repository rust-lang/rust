// Test that we don't duplicate storage for a variable that is moved to another
// binding. This used to happen in the presence of unwind and drop edges (see
// `complex` below.)
//
// The exact sizes here can change (we'd like to know when they do). What we
// don't want to see is the `complex` generator size being upwards of 2048 bytes
// (which would indicate it is reserving space for two copies of Foo.)
//
// See issue #59123 for a full explanation.

// edition:2018

#![feature(generators, generator_trait)]

use std::ops::Generator;

const FOO_SIZE: usize = 1024;
struct Foo([u8; FOO_SIZE]);

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn simple() -> impl Generator<Yield = (), Return = ()> {
    static || {
        let first = Foo([0; FOO_SIZE]);
        let _second = first;
        yield;
        // _second dropped here
    }
}

fn noop() {}

fn complex() -> impl Generator<Yield = (), Return = ()> {
    static || {
        let first = Foo([0; FOO_SIZE]);
        noop();
        let _second = first;
        yield;
        // _second dropped here
    }
}

fn main() {
    assert_eq!(1028, std::mem::size_of_val(&simple()));
    assert_eq!(1032, std::mem::size_of_val(&complex()));
}
