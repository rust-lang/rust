use super::*;
use crate::io::{Read, Write};

#[test]
fn pipe_creation_and_rw() {
    let (mut rx, mut tx) = pipe().unwrap();
    tx.write_all(b"12345").unwrap();
    drop(tx);

    let mut s = String::new();
    rx.read_to_string(&mut s).unwrap();
    assert_eq!(s, "12345");
}

#[test]
fn pipe_try_clone_and_rw() {
    let (mut rx, mut tx) = pipe().unwrap();
    tx.try_clone().unwrap().write_all(b"12").unwrap();
    tx.write_all(b"345").unwrap();
    drop(tx);

    let mut s = String::new();
    rx.try_clone().unwrap().take(3).read_to_string(&mut s).unwrap();
    assert_eq!(s, "123");

    s.clear();
    rx.read_to_string(&mut s).unwrap();
    assert_eq!(s, "45");
}
