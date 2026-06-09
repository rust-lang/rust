//@ run-pass
// A smoke test for recursive enum structures using Box<T>.
// This test constructs a linked list-like structure to exercise memory allocation and ownership.
// Originally introduced in 2010, this is one of Rust’s earliest test cases.

#![allow(dead_code)]

enum List {
    Cons(isize, Box<List>),
    Nil,
}

fn main() {
    List::Cons(
        10,
        Box::new(List::Cons(
            11,
            Box::new(List::Cons(12, Box::new(List::Nil))),
        )),
    );
}
