//@ run-pass
// Test that we don't duplicate storage for a variable that is moved to another
// binding. This used to happen in the presence of unwind and drop edges (see
// `complex` below.)
//
// The exact sizes here can change (we'd like to know when they do). What we
// don't want to see is the `complex` coroutine size being upwards of 2048 bytes
// (which would indicate it is reserving space for two copies of Foo.)
//
// See issue #59123 for a full explanation.

//@ edition:2018
//@ needs-unwind Size of Closures change on panic=abort

#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;

const FOO_SIZE: usize = 1024;
struct Foo(#[allow(dead_code)] [u8; FOO_SIZE]);

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn move_before_yield() -> impl Coroutine<Yield = (), Return = ()> {
    #[coroutine]
    static || {
        let first = Foo([0; FOO_SIZE]);
        let _second = first;
        yield;
        // _second dropped here
    }
}

fn noop() {}

fn move_before_yield_with_noop() -> impl Coroutine<Yield = (), Return = ()> {
    #[coroutine]
    static || {
        let first = Foo([0; FOO_SIZE]);
        noop();
        let _second = first;
        yield;
        // _second dropped here
    }
}

// Today we don't have NRVO (we allocate space for both `first` and `second`,)
// but we can overlap `first` with `_third`.
fn overlap_move_points() -> impl Coroutine<Yield = (), Return = ()> {
    #[coroutine]
    static || {
        let first = Foo([0; FOO_SIZE]);
        yield;
        let second = first;
        yield;
        let _third = second;
        yield;
    }
}

fn overlap_x_and_y() -> impl Coroutine<Yield = (), Return = ()> {
    #[coroutine]
    static || {
        let x = Foo([0; FOO_SIZE]);
        yield;
        drop(x);
        let y = Foo([0; FOO_SIZE]);
        yield;
        drop(y);
    }
}

fn main() {
    assert_eq!(1025, std::mem::size_of_val(&move_before_yield()));
    assert_eq!(1026, std::mem::size_of_val(&move_before_yield_with_noop()));
    assert_eq!(2051, std::mem::size_of_val(&overlap_move_points()));
    assert_eq!(1026, std::mem::size_of_val(&overlap_x_and_y()));
}
