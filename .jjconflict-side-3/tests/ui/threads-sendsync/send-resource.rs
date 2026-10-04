//@ run-pass
#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(non_camel_case_types)]

//@ needs-threads

use std::sync::mpsc::channel;
use std::thread;

struct test {
    f: isize,
}

impl Drop for test {
    fn drop(&mut self) {}
}

fn test(f: isize) -> test {
    test { f: f }
}

pub fn main() {
    let (tx, rx) = channel();

    let t = thread::spawn(move || {
        let (tx2, rx2) = channel();
        tx.send(tx2).unwrap();

        let _r = rx2.recv().unwrap();
    });

    rx.recv().unwrap().send(test(42)).unwrap();

    t.join();
}
