//@ run-pass

#![allow(unused_must_use)]
//@ needs-threads

use std::thread;

struct Pair {
    a: isize,
    b: isize,
}

pub fn main() {
    let z: Box<_> = Box::new(Pair { a: 10, b: 12 });

    thread::spawn(move || {
        assert_eq!(z.a, 10);
        assert_eq!(z.b, 12);
    })
    .join();
}
