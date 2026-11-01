// https://github.com/rust-lang/rust/issues/29710
//@ check-pass
#![deny(unused_results)]
#![allow(dead_code)]

#[derive(Debug)]
struct A(usize);

#[derive(Debug)]
struct B { a: usize }

fn main() {}
