// ignore-windows: Concurrency on Windows is not supported yet.

//! Check if Rust channels are working.

use std::sync::mpsc::{channel, sync_channel};
use std::thread;

/// The test taken from the Rust documentation.
fn simple_send() {
    let (tx, rx) = channel();
    thread::spawn(move || {
        tx.send(10).unwrap();
    });
    assert_eq!(rx.recv().unwrap(), 10);
}

/// The test taken from the Rust documentation.
fn multiple_send() {
    let (tx, rx) = channel();
    for i in 0..10 {
        let tx = tx.clone();
        thread::spawn(move || {
            tx.send(i).unwrap();
        });
    }

    let mut sum = 0;
    for _ in 0..10 {
        let j = rx.recv().unwrap();
        assert!(0 <= j && j < 10);
        sum += j;
    }
    assert_eq!(sum, 45);
}

/// The test taken from the Rust documentation.
fn send_on_sync() {
    let (sender, receiver) = sync_channel(1);

    // this returns immediately
    sender.send(1).unwrap();

    thread::spawn(move || {
        // this will block until the previous message has been received
        sender.send(2).unwrap();
    });

    assert_eq!(receiver.recv().unwrap(), 1);
    assert_eq!(receiver.recv().unwrap(), 2);
}

fn main() {
    simple_send();
    multiple_send();
    send_on_sync();
}
