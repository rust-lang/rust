//@ run-pass
#![allow(unused_mut)]
#![allow(unused_parens)]
#![allow(non_camel_case_types)]

use std::cmp;
use std::sync::mpsc::channel;

// Tests of ports and channels on various types
fn test_rec() {
    struct R {
        val0: isize,
        val1: u8,
        val2: char,
    }

    let (tx, rx) = channel();
    let r0: R = R { val0: 0, val1: 1, val2: '2' };
    tx.send(r0).unwrap();
    let mut r1: R;
    r1 = rx.recv().unwrap();
    assert_eq!(r1.val0, 0);
    assert_eq!(r1.val1, 1);
    assert_eq!(r1.val2, '2');
}

fn test_vec() {
    let (tx, rx) = channel();
    let v0: Vec<isize> = vec![0, 1, 2];
    tx.send(v0).unwrap();
    let v1 = rx.recv().unwrap();
    assert_eq!(v1[0], 0);
    assert_eq!(v1[1], 1);
    assert_eq!(v1[2], 2);
}

fn test_str() {
    let (tx, rx) = channel();
    let s0 = "test".to_string();
    tx.send(s0).unwrap();
    let s1 = rx.recv().unwrap();
    assert_eq!(s1.as_bytes()[0], 't' as u8);
    assert_eq!(s1.as_bytes()[1], 'e' as u8);
    assert_eq!(s1.as_bytes()[2], 's' as u8);
    assert_eq!(s1.as_bytes()[3], 't' as u8);
}

#[derive(Debug)]
enum t {
    tag1,
    tag2(isize),
    tag3(isize, u8, char),
}

impl cmp::PartialEq for t {
    fn eq(&self, other: &t) -> bool {
        match *self {
            t::tag1 => match (*other) {
                t::tag1 => true,
                _ => false,
            },
            t::tag2(e0a) => match (*other) {
                t::tag2(e0b) => e0a == e0b,
                _ => false,
            },
            t::tag3(e0a, e1a, e2a) => match (*other) {
                t::tag3(e0b, e1b, e2b) => e0a == e0b && e1a == e1b && e2a == e2b,
                _ => false,
            },
        }
    }
    fn ne(&self, other: &t) -> bool {
        !(*self).eq(other)
    }
}

fn test_tag() {
    let (tx, rx) = channel();
    tx.send(t::tag1).unwrap();
    tx.send(t::tag2(10)).unwrap();
    tx.send(t::tag3(10, 11, 'A')).unwrap();
    let mut t1: t;
    t1 = rx.recv().unwrap();
    assert_eq!(t1, t::tag1);
    t1 = rx.recv().unwrap();
    assert_eq!(t1, t::tag2(10));
    t1 = rx.recv().unwrap();
    assert_eq!(t1, t::tag3(10, 11, 'A'));
}

fn test_chan() {
    let (tx1, rx1) = channel();
    let (tx2, rx2) = channel();
    tx1.send(tx2).unwrap();
    let tx2 = rx1.recv().unwrap();
    // Does the transmitted channel still work?

    tx2.send(10).unwrap();
    let mut i: isize;
    i = rx2.recv().unwrap();
    assert_eq!(i, 10);
}

pub fn main() {
    test_rec();
    test_vec();
    test_str();
    test_tag();
    test_chan();
}
