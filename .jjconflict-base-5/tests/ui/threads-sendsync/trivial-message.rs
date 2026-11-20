//@ run-pass

#![allow(unused_must_use)]
/*
 This is about the simplest program that can successfully send a
 message.
*/

use std::sync::mpsc::channel;

pub fn main() {
    let (tx, rx) = channel();
    tx.send(42);
    let r = rx.recv();
    println!("{:?}", r);
}
