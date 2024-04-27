//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-strict-provenance

use std::sync::mpsc::{channel, sync_channel};
use std::thread;

// Check if channels are working.

/// The test taken from the Rust documentation.
fn simple_send() {
    let (tx, rx) = channel();
    let t = thread::spawn(move || {
        tx.send(10).unwrap();
    });
    assert_eq!(rx.recv().unwrap(), 10);
    t.join().unwrap();
}

/// The test taken from the Rust documentation.
fn multiple_send() {
    let (tx, rx) = channel();
    let mut threads = vec![];
    for i in 0..10 {
        let tx = tx.clone();
        let t = thread::spawn(move || {
            tx.send(i).unwrap();
        });
        threads.push(t);
    }

    let mut sum = 0;
    for _ in 0..10 {
        let j = rx.recv().unwrap();
        assert!(0 <= j && j < 10);
        sum += j;
    }
    assert_eq!(sum, 45);

    for t in threads {
        t.join().unwrap();
    }
}

/// The test taken from the Rust documentation.
fn send_on_sync() {
    let (sender, receiver) = sync_channel(1);

    // this returns immediately
    sender.send(1).unwrap();

    let t = thread::spawn(move || {
        // this will block until the previous message has been received
        sender.send(2).unwrap();
    });

    assert_eq!(receiver.recv().unwrap(), 1);
    assert_eq!(receiver.recv().unwrap(), 2);

    t.join().unwrap();
}

fn main() {
    simple_send();
    multiple_send();
    send_on_sync();
}
