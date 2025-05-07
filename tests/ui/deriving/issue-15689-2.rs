//@ check-pass
#![allow(dead_code)]

#[derive(Clone)]
enum Test<'a> {
    Slice(&'a isize)
}

fn main() {}
