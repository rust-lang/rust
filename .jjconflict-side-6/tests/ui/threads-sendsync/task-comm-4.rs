//@ run-pass
#![allow(unused_assignments)]

use std::sync::mpsc::channel;

pub fn main() {
    test00();
}

fn test00() {
    let mut r: isize = 0;
    let mut sum: isize = 0;
    let (tx, rx) = channel();
    tx.send(1).unwrap();
    tx.send(2).unwrap();
    tx.send(3).unwrap();
    tx.send(4).unwrap();
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    tx.send(5).unwrap();
    tx.send(6).unwrap();
    tx.send(7).unwrap();
    tx.send(8).unwrap();
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    assert_eq!(sum, 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
}
