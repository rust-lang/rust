#![allow(non_camel_case_types)]
// ignore-emscripten no threads support

use std::sync::mpsc::{channel, Sender};
use std::thread;

struct complainer {
    tx: Sender<bool>,
}

impl Drop for complainer {
    fn drop(&mut self) {
        println!("About to send!");
        self.tx.send(true).unwrap();
        println!("Sent!");
    }
}

fn complainer(tx: Sender<bool>) -> complainer {
    println!("Hello!");
    complainer {
        tx: tx
    }
}

fn f(tx: Sender<bool>) {
    let _tx = complainer(tx);
    panic!();
}

pub fn main() {
    let (tx, rx) = channel();
    let t = thread::spawn(move|| f(tx.clone()));
    println!("hiiiiiiiii");
    assert!(rx.recv().unwrap());
    drop(t.join());
}
