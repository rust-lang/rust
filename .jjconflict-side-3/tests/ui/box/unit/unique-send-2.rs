//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

use std::sync::mpsc::{channel, Sender};
use std::thread;

fn child(tx: &Sender<Box<usize>>, i: usize) {
    tx.send(Box::new(i)).unwrap();
}

pub fn main() {
    let (tx, rx) = channel();
    let n = 100;
    let mut expected = 0;
    let ts = (0..n).map(|i| {
        expected += i;
        let tx = tx.clone();
        thread::spawn(move|| {
            child(&tx, i)
        })
    }).collect::<Vec<_>>();

    let mut actual = 0;
    for _ in 0..n {
        let j = rx.recv().unwrap();
        actual += *j;
    }

    assert_eq!(expected, actual);

    for t in ts { t.join(); }
}
