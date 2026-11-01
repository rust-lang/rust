// Test that a class with only sendable fields can be sent

use std::sync::mpsc::channel;

#[allow(dead_code)]
struct Foo {
    i: isize,
    j: char,
}

fn foo(i: isize, j: char) -> Foo {
    Foo { i: i, j: j }
}

fn main() {
    let (tx, rx) = channel();
    tx.send(foo(42, 'c')).unwrap();
    let _val = rx;
}
