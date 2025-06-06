//@ run-pass
//@ needs-unwind

#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(unused_variables)]
// Test cleanup of rvalue temporary that occurs while `box` construction
// is in progress. This scenario revealed a rather terrible bug.  The
// ingredients are:
//
// 1. Partial cleanup of `box` is in scope,
// 2. cleanup of return value from `get_bar()` is in scope,
// 3. do_it() panics.
//
// This led to a bug because `the top-most frame that was to be
// cleaned (which happens to be the partial cleanup of `box`) required
// multiple basic blocks, which led to us dropping part of the cleanup
// from the top-most frame.
//
// It's unclear how likely such a bug is to recur, but it seems like a
// scenario worth testing.

//@ needs-threads

use std::thread;

enum Conzabble {
    Bickwick(Foo),
}

struct Foo {
    field: Box<usize>,
}

fn do_it(x: &[usize]) -> Foo {
    panic!()
}

fn get_bar(x: usize) -> Vec<usize> {
    vec![x * 2]
}

pub fn fails() {
    let x = 2;
    let mut y: Vec<Box<_>> = Vec::new();
    y.push(Box::new(Conzabble::Bickwick(do_it(&get_bar(x)))));
}

pub fn main() {
    thread::spawn(fails).join();
}
