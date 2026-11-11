//@ run-pass
//@ needs-threads

use std::sync::mpsc::channel;
use std::thread;

pub fn main() {
    let (tx, rx) = channel::<&'static str>();

    let t = thread::spawn(move || {
        assert_eq!(rx.recv().unwrap(), "hello, world");
    });

    tx.send("hello, world").unwrap();
    t.join().ok().unwrap();
}
