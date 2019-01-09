#![warn(clippy::needless_update)]
#![allow(clippy::no_effect)]

struct S {
    pub a: i32,
    pub b: i32,
}

fn main() {
    let base = S { a: 0, b: 0 };
    S { ..base }; // no error
    S { a: 1, ..base }; // no error
    S { a: 1, b: 1, ..base };
}
