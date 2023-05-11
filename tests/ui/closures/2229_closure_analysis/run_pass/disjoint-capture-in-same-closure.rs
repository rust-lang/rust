// edition:2021
// run-pass

// Tests that if a closure uses individual fields of the same object
// then that case is handled properly.

#![allow(unused)]

struct Struct {
    x: i32,
    y: i32,
    s: String,
}

fn main() {
    let mut s = Struct { x: 10, y: 10, s: String::new() };

    let mut c = {
        s.x += 10;
        s.y += 42;
        s.s = String::from("new");
    };
}
