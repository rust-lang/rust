#![feature(box_syntax)]

use std::sync::mpsc::channel;

pub fn main() {
    let (tx, rx) = channel::<Box<_>>();
    tx.send(box 100).unwrap();
    let v = rx.recv().unwrap();
    assert_eq!(v, box 100);

    tx.send(box 101).unwrap();
    tx.send(box 102).unwrap();
    assert_eq!(rx.recv().unwrap(), box 101);
    assert_eq!(rx.recv().unwrap(), box 102);
}
