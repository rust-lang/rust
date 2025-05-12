//@ known-bug: #122630
//@ compile-flags: -Zvalidate-mir

use std::ops::Coroutine;

const FOO_SIZE: usize = 1024;
struct Foo([u8; FOO_SIZE]);

impl Drop for Foo {
    fn move_before_yield_with_noop() -> impl Coroutine<Yield = ()> {}
}

fn overlap_move_points() -> impl Coroutine<Yield = ()> {
    static || {
        let first = Foo([0; FOO_SIZE]);
        yield;
        let second = first;
        yield;
        let second = first;
        yield;
    }
}
