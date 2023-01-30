// run-pass
#![allow(unused_variables)]
// ignore-emscripten no threads support

use std::sync::mpsc::{channel, Sender};
use std::thread;

fn start(tx: &Sender<isize>, start: isize, number_of_messages: isize) {
    let mut i: isize = 0;
    while i< number_of_messages { tx.send(start + i).unwrap(); i += 1; }
}

pub fn main() {
    println!("Check that we don't deadlock.");
    let (tx, rx) = channel();
    let _ = thread::spawn(move|| { start(&tx, 0, 10) }).join();
    println!("Joined task");
}
