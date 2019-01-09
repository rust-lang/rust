// run-pass
#![allow(unused_must_use)]
// ignore-emscripten no threads support

use std::thread;
use std::sync::mpsc::{channel, Sender};

fn producer(tx: &Sender<Vec<u8>>) {
    tx.send(
         vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
          13]).unwrap();
}

pub fn main() {
    let (tx, rx) = channel::<Vec<u8>>();
    let prod = thread::spawn(move|| {
        producer(&tx)
    });

    let _data: Vec<u8> = rx.recv().unwrap();
    prod.join();
}
