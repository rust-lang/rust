// run-pass
#![allow(unused_must_use)]
// ignore-emscripten no threads support

// This test may not always fail, but it can be flaky if the race it used to
// expose is still present.

#![feature(mpsc_select)]
#![allow(deprecated)]

use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread;

fn helper(rx: Receiver<Sender<()>>) {
    for tx in rx.iter() {
        let _ = tx.send(());
    }
}

fn main() {
    let (tx, rx) = channel();
    let t = thread::spawn(move|| { helper(rx) });
    let (snd, rcv) = channel::<isize>();
    for _ in 1..100000 {
        snd.send(1).unwrap();
        let (tx2, rx2) = channel();
        tx.send(tx2).unwrap();
        select! {
            _ = rx2.recv() => (),
            _ = rcv.recv() => ()
        }
    }
    drop(tx);
    t.join();
}
