use crate::io::prelude::*;
use crate::io::{copy, empty, repeat, sink, Empty, Repeat, Sink};

#[test]
fn copy_copies() {
    let mut r = repeat(0).take(4);
    let mut w = sink();
    assert_eq!(copy(&mut r, &mut w).unwrap(), 4);

    let mut r = repeat(0).take(1 << 17);
    assert_eq!(copy(&mut r as &mut dyn Read, &mut w as &mut dyn Write).unwrap(), 1 << 17);
}

#[test]
fn sink_sinks() {
    let mut s = sink();
    assert_eq!(s.write(&[]).unwrap(), 0);
    assert_eq!(s.write(&[0]).unwrap(), 1);
    assert_eq!(s.write(&[0; 1024]).unwrap(), 1024);
    assert_eq!(s.by_ref().write(&[0; 1024]).unwrap(), 1024);
}

#[test]
fn empty_reads() {
    let mut e = empty();
    assert_eq!(e.read(&mut []).unwrap(), 0);
    assert_eq!(e.read(&mut [0]).unwrap(), 0);
    assert_eq!(e.read(&mut [0; 1024]).unwrap(), 0);
    assert_eq!(e.by_ref().read(&mut [0; 1024]).unwrap(), 0);
}

#[test]
fn repeat_repeats() {
    let mut r = repeat(4);
    let mut b = [0; 1024];
    assert_eq!(r.read(&mut b).unwrap(), 1024);
    assert!(b.iter().all(|b| *b == 4));
}

#[test]
fn take_some_bytes() {
    assert_eq!(repeat(4).take(100).bytes().count(), 100);
    assert_eq!(repeat(4).take(100).bytes().next().unwrap().unwrap(), 4);
    assert_eq!(repeat(1).take(10).chain(repeat(2).take(10)).bytes().count(), 20);
}

#[allow(dead_code)]
fn const_utils() {
    const _: Empty = empty();
    const _: Repeat = repeat(b'c');
    const _: Sink = sink();
}
