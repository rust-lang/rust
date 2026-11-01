//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

use std::sync::mpsc::{channel, Sender};
use std::thread;

pub fn main() {
    test05();
}

fn test05_start(tx: &Sender<isize>) {
    tx.send(10).unwrap();
    println!("sent 10");
    tx.send(20).unwrap();
    println!("sent 20");
    tx.send(30).unwrap();
    println!("sent 30");
}

fn test05() {
    let (tx, rx) = channel();
    let t = thread::spawn(move || test05_start(&tx));
    let mut value: isize = rx.recv().unwrap();
    println!("{}", value);
    value = rx.recv().unwrap();
    println!("{}", value);
    value = rx.recv().unwrap();
    println!("{}", value);
    assert_eq!(value, 30);
    t.join();
}
