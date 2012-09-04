// -*- rust -*-

use std;
import pipes;
import pipes::send;
import pipes::Port;
import pipes::recv;
import pipes::Chan;

// Tests of ports and channels on various types
fn test_rec() {
    type r = {val0: int, val1: u8, val2: char};

    let (ch, po) = pipes::stream();
    let r0: r = {val0: 0, val1: 1u8, val2: '2'};
    ch.send(r0);
    let mut r1: r;
    r1 = po.recv();
    assert (r1.val0 == 0);
    assert (r1.val1 == 1u8);
    assert (r1.val2 == '2');
}

fn test_vec() {
    let (ch, po) = pipes::stream();
    let v0: ~[int] = ~[0, 1, 2];
    ch.send(v0);
    let v1 = po.recv();
    assert (v1[0] == 0);
    assert (v1[1] == 1);
    assert (v1[2] == 2);
}

fn test_str() {
    let (ch, po) = pipes::stream();
    let s0 = ~"test";
    ch.send(s0);
    let s1 = po.recv();
    assert (s1[0] == 't' as u8);
    assert (s1[1] == 'e' as u8);
    assert (s1[2] == 's' as u8);
    assert (s1[3] == 't' as u8);
}

enum t {
    tag1,
    tag2(int),
    tag3(int, u8, char)
}

impl t : cmp::Eq {
    pure fn eq(&&other: t) -> bool {
        match self {
            tag1 => {
                match other {
                    tag1 => true,
                    _ => false
                }
            }
            tag2(e0a) => {
                match other {
                    tag2(e0b) => e0a == e0b,
                    _ => false
                }
            }
            tag3(e0a, e1a, e2a) => {
                match other {
                    tag3(e0b, e1b, e2b) =>
                        e0a == e0b && e1a == e1b && e2a == e2b,
                    _ => false
                }
            }
        }
    }
}

fn test_tag() {
    let (ch, po) = pipes::stream();
    ch.send(tag1);
    ch.send(tag2(10));
    ch.send(tag3(10, 11u8, 'A'));
    let mut t1: t;
    t1 = po.recv();
    assert (t1 == tag1);
    t1 = po.recv();
    assert (t1 == tag2(10));
    t1 = po.recv();
    assert (t1 == tag3(10, 11u8, 'A'));
}

fn test_chan() {
    let (ch, po) = pipes::stream();
    let (ch0, po0) = pipes::stream();
    ch.send(ch0);
    let ch1 = po.recv();
    // Does the transmitted channel still work?

    ch1.send(10);
    let mut i: int;
    i = po0.recv();
    assert (i == 10);
}

fn main() {
    test_rec();
    test_vec();
    test_str();
    test_tag();
    test_chan();
}
