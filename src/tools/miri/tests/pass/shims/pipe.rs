//@ignore-target: windows

#![feature(anonymous_pipe)]

use std::io::{Read, Write, pipe};

fn main() {
    let (mut ping_rx, mut ping_tx) = pipe().unwrap();
    ping_tx.write(b"hello").unwrap();
    let mut buf: [u8; 5] = [0; 5];
    ping_rx.read(&mut buf).unwrap();
    assert_eq!(&buf, "hello".as_bytes());
}
