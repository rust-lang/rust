//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

use std::sync::mpsc::{channel, Sender};
use std::thread;

fn start(tx: &Sender<isize>, i0: isize) {
    let mut i = i0;
    while i > 0 {
        tx.send(0).unwrap();
        i = i - 1;
    }
}

pub fn main() {
    // Spawn a thread that sends us back messages. The parent thread
    // is likely to terminate before the child completes, so from
    // the child's point of view the receiver may die. We should
    // drop messages on the floor in this case, and not crash!
    let (tx, rx) = channel();
    let t = thread::spawn(move || start(&tx, 10));
    rx.recv();
    t.join();
}
