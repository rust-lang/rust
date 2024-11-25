//@ run-pass
#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_camel_case_types)]

// Test that a class with only sendable fields can be sent


use std::sync::mpsc::channel;

struct foo {
    i: isize,
    j: char,
}

fn foo(i: isize, j: char) -> foo {
    foo { i: i, j: j }
}

pub fn main() {
    let (tx, rx) = channel();
    tx.send(foo(42, 'c'));
}
