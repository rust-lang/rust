//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

use std::sync::mpsc::{channel, Sender};
use std::thread;

fn start(tx: &Sender<Sender<isize>>) {
    let (tx2, _rx) = channel();
    tx.send(tx2).unwrap();
}

pub fn main() {
    let (tx, rx) = channel();
    let child = thread::spawn(move || start(&tx));
    let _tx = rx.recv().unwrap();
    child.join();
}
