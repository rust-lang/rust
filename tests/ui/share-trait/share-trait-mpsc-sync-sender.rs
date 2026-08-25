//@ run-pass

#![feature(share_trait)]

use std::clone::Share;
use std::sync::mpsc::sync_channel;

fn main() {
    let (sender, receiver) = sync_channel(2);
    let shared_sender = sender.share();

    sender.send(1).unwrap();
    shared_sender.send(2).unwrap();

    let mut received = [receiver.recv().unwrap(), receiver.recv().unwrap()];
    received.sort();

    assert_eq!(received, [1, 2]);
}
