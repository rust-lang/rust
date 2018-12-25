#![allow(unused_must_use)]
// ignore-emscripten no threads support

#![feature(box_syntax)]

use std::thread;

struct Pair {
    a: isize,
    b: isize
}

pub fn main() {
    let z: Box<_> = box Pair { a : 10, b : 12};

    thread::spawn(move|| {
        assert_eq!(z.a, 10);
        assert_eq!(z.b, 12);
    }).join();
}
