// run-pass
#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616
// ignore-emscripten no threads support

use std::thread;
use std::sync::mpsc::channel;

struct test {
  f: isize,
}

impl Drop for test {
    fn drop(&mut self) {}
}

fn test(f: isize) -> test {
    test {
        f: f
    }
}

pub fn main() {
    let (tx, rx) = channel();

    let t = thread::spawn(move|| {
        let (tx2, rx2) = channel();
        tx.send(tx2).unwrap();

        let _r = rx2.recv().unwrap();
    });

    rx.recv().unwrap().send(test(42)).unwrap();

    t.join();
}
