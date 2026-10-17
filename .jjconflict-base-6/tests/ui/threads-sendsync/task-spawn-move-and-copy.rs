//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

use std::sync::mpsc::channel;
use std::thread;

pub fn main() {
    let (tx, rx) = channel::<usize>();

    let x: Box<isize> = Box::new(1);
    let x_in_parent = &(*x) as *const isize as usize;

    let t = thread::spawn(move || {
        let x_in_child = &(*x) as *const isize as usize;
        tx.send(x_in_child).unwrap();
    });

    let x_in_child = rx.recv().unwrap();
    assert_eq!(x_in_parent, x_in_child);

    t.join();
}
