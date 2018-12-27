// run-pass
#![feature(box_syntax)]

use std::sync::mpsc::channel;

pub fn main() {
    let (tx, rx) = channel::<Box<_>>();
    tx.send(box 100).unwrap();
    let v = rx.recv().unwrap();
    assert_eq!(v, box 100);
}
