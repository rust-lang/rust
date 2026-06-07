//! Regression test for https://github.com/rust-lang/rust/issues/23491

//@ run-pass
#![allow(unused_variables)]

struct Node<T: ?Sized>(#[allow(dead_code)] T);

fn main() {
    let x: Box<Node<[isize]>> = Box::new(Node([]));
}
