//! regression test for <https://github.com/rust-lang/rust/issues/25394>
//@ check-pass
#![allow(dead_code)]
#[derive(Debug)]
struct Row<T>([T]);

fn main() {}
