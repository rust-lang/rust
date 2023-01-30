// run-pass
#![allow(unused_parens)]
// ignore-emscripten no threads support

use std::sync::mpsc::{channel, Sender};
use std::thread;

pub fn main() {
    let (tx, rx) = channel();

    // Spawn 10 threads each sending us back one isize.
    let mut i = 10;
    while (i > 0) {
        println!("{}", i);
        let tx = tx.clone();
        thread::spawn({let i = i; move|| { child(i, &tx) }});
        i = i - 1;
    }

    // Spawned threads are likely killed before they get a chance to send
    // anything back, so we deadlock here.

    i = 10;
    while (i > 0) {
        println!("{}", i);
        rx.recv().unwrap();
        i = i - 1;
    }

    println!("main thread exiting");
}

fn child(x: isize, tx: &Sender<isize>) {
    println!("{}", x);
    tx.send(x).unwrap();
}
