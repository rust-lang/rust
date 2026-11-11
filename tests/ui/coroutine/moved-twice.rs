//! Regression test for #122630
//@ compile-flags: -Zvalidate-mir

#![feature(coroutines, coroutine_trait, yield_expr)]

use std::ops::Coroutine;

const FOO_SIZE: usize = 1024;
struct Foo([u8; FOO_SIZE]);

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn overlap_move_points() -> impl Coroutine<Yield = ()> {
    #[coroutine] static || {
        let first = Foo([0; FOO_SIZE]);
        yield;
        let second = first;
        yield;
        let second = first;
        //~^ ERROR: use of moved value: `first` [E0382]
        yield;
    }
}

fn main() {}
