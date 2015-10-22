#![feature(plugin)]
#![plugin(clippy)]

#![deny(needless_update)]

struct S {
    pub a: i32,
    pub b: i32,
}

fn main() {
    let base = S { a: 0, b: 0 };
    S { ..base }; // no error
    S { a: 1, ..base }; // no error
    S { a: 1, b: 1, ..base }; //~ERROR struct update has no effect
}
