//@ run-pass
// A smoke test for recursive enum structures using Box<T>.
// This test constructs a linked list-like structure to verify memory allocation and ownership.
// Originally introduced in 2010, this is one of Rust’s earliest test cases.

#![allow(non_camel_case_types)]

enum list { #[allow(dead_code)] cons(isize, Box<list>), nil, }

pub fn main() {
    list::cons(10, Box::new(list::cons(11, Box::new(list::cons(12, Box::new(list::nil))))));
}
