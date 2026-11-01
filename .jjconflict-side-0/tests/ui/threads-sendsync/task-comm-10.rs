//@ run-pass
#![allow(unused_must_use)]
#![allow(unused_mut)]
//@ needs-threads

use std::sync::mpsc::{channel, Sender};
use std::thread;

fn start(tx: &Sender<Sender<String>>) {
    let (tx2, rx) = channel();
    tx.send(tx2).unwrap();

    let mut a;
    let mut b;
    a = rx.recv().unwrap();
    assert_eq!(a, "A".to_string());
    println!("{}", a);
    b = rx.recv().unwrap();
    assert_eq!(b, "B".to_string());
    println!("{}", b);
}

pub fn main() {
    let (tx, rx) = channel();
    let child = thread::spawn(move || start(&tx));

    let mut c = rx.recv().unwrap();
    c.send("A".to_string()).unwrap();
    c.send("B".to_string()).unwrap();
    thread::yield_now();

    child.join();
}
