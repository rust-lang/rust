//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

use std::sync::mpsc::{channel, Sender};
use std::thread;

pub fn main() {
    let (tx, rx) = channel();
    let t = thread::spawn(move || child(&tx));
    let y = rx.recv().unwrap();
    println!("received");
    println!("{}", y);
    assert_eq!(y, 10);
    t.join();
}

fn child(c: &Sender<isize>) {
    println!("sending");
    c.send(10).unwrap();
    println!("value sent");
}
