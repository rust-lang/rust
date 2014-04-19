// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;

// Tests of ports and channels on various types
fn test_rec() {
    struct R {val0: int, val1: u8, val2: char}

    let (tx, rx) = channel();
    let r0: R = R {val0: 0, val1: 1u8, val2: '2'};
    tx.send(r0);
    let mut r1: R;
    r1 = rx.recv();
    assert_eq!(r1.val0, 0);
    assert_eq!(r1.val1, 1u8);
    assert_eq!(r1.val2, '2');
}

fn test_vec() {
    let (tx, rx) = channel();
    let v0: Vec<int> = vec!(0, 1, 2);
    tx.send(v0);
    let v1 = rx.recv();
    assert_eq!(*v1.get(0), 0);
    assert_eq!(*v1.get(1), 1);
    assert_eq!(*v1.get(2), 2);
}

fn test_str() {
    let (tx, rx) = channel();
    let s0 = "test".to_owned();
    tx.send(s0);
    let s1 = rx.recv();
    assert_eq!(s1[0], 't' as u8);
    assert_eq!(s1[1], 'e' as u8);
    assert_eq!(s1[2], 's' as u8);
    assert_eq!(s1[3], 't' as u8);
}

#[deriving(Show)]
enum t {
    tag1,
    tag2(int),
    tag3(int, u8, char)
}

impl cmp::Eq for t {
    fn eq(&self, other: &t) -> bool {
        match *self {
            tag1 => {
                match (*other) {
                    tag1 => true,
                    _ => false
                }
            }
            tag2(e0a) => {
                match (*other) {
                    tag2(e0b) => e0a == e0b,
                    _ => false
                }
            }
            tag3(e0a, e1a, e2a) => {
                match (*other) {
                    tag3(e0b, e1b, e2b) =>
                        e0a == e0b && e1a == e1b && e2a == e2b,
                    _ => false
                }
            }
        }
    }
    fn ne(&self, other: &t) -> bool { !(*self).eq(other) }
}

fn test_tag() {
    let (tx, rx) = channel();
    tx.send(tag1);
    tx.send(tag2(10));
    tx.send(tag3(10, 11u8, 'A'));
    let mut t1: t;
    t1 = rx.recv();
    assert_eq!(t1, tag1);
    t1 = rx.recv();
    assert_eq!(t1, tag2(10));
    t1 = rx.recv();
    assert_eq!(t1, tag3(10, 11u8, 'A'));
}

fn test_chan() {
    let (tx1, rx1) = channel();
    let (tx2, rx2) = channel();
    tx1.send(tx2);
    let tx2 = rx1.recv();
    // Does the transmitted channel still work?

    tx2.send(10);
    let mut i: int;
    i = rx2.recv();
    assert_eq!(i, 10);
}

pub fn main() {
    test_rec();
    test_vec();
    test_str();
    test_tag();
    test_chan();
}
