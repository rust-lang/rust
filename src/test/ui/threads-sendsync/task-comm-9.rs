// run-pass
#![allow(unused_must_use)]
// ignore-emscripten no threads support

use std::thread;
use std::sync::mpsc::{channel, Sender};

pub fn main() { test00(); }

fn test00_start(c: &Sender<isize>, number_of_messages: isize) {
    let mut i: isize = 0;
    while i < number_of_messages { c.send(i + 0).unwrap(); i += 1; }
}

fn test00() {
    let r: isize = 0;
    let mut sum: isize = 0;
    let (tx, rx) = channel();
    let number_of_messages: isize = 10;

    let result = thread::spawn(move|| {
        test00_start(&tx, number_of_messages);
    });

    let mut i: isize = 0;
    while i < number_of_messages {
        sum += rx.recv().unwrap();
        println!("{}", r);
        i += 1;
    }

    result.join();

    assert_eq!(sum, number_of_messages * (number_of_messages - 1) / 2);
}
