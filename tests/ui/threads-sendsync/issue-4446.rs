//@ run-pass
//@ needs-threads

use std::sync::mpsc::channel;
use std::thread;

pub fn main() {
    let (tx, rx) = channel();

    tx.send("hello, world").unwrap();

    thread::spawn(move || {
        println!("{}", rx.recv().unwrap());
    })
    .join()
    .ok()
    .unwrap();
}
