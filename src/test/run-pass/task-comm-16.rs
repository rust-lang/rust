// -*- rust -*-

use std;
import std::comm;
import std::comm::mk_port;

// Tests of ports and channels on various types
fn test_rec() {
    type r = {val0: int, val1: u8, val2: char};

    let po = comm::mk_port();
    let ch = po.mk_chan();
    let r0: r = {val0: 0, val1: 1u8, val2: '2'};
    ch.send(r0);
    let r1: r;
    r1 = po.recv();
    assert (r1.val0 == 0);
    assert (r1.val1 == 1u8);
    assert (r1.val2 == '2');
}

fn test_vec() {
    let po = comm::mk_port();
    let ch = po.mk_chan();
    let v0: [int] = ~[0, 1, 2];
    ch.send(v0);
    let v1: [int];
    v1 = po.recv();
    assert (v1.(0) == 0);
    assert (v1.(1) == 1);
    assert (v1.(2) == 2);
}

fn test_str() {
    let po = comm::mk_port();
    let ch = po.mk_chan();
    let s0: str = "test";
    ch.send(s0);
    let s1: str;
    s1 = po.recv();
    assert (s1.(0) as u8 == 't' as u8);
    assert (s1.(1) as u8 == 'e' as u8);
    assert (s1.(2) as u8 == 's' as u8);
    assert (s1.(3) as u8 == 't' as u8);
}

fn test_tag() {
    tag t { tag1; tag2(int); tag3(int, u8, char); }
    let po = comm::mk_port();
    let ch = po.mk_chan();
    ch.send(tag1);
    ch.send(tag2(10));
    ch.send(tag3(10, 11u8, 'A'));
    // FIXME: Do port semantics really guarantee these happen in order?
    let t1: t;
    t1 = po.recv();
    assert (t1 == tag1);
    t1 = po.recv();
    assert (t1 == tag2(10));
    t1 = po.recv();
    assert (t1 == tag3(10, 11u8, 'A'));
}

fn test_chan() {
    let po = comm::mk_port();
    let ch = po.mk_chan();
    let po0 = comm::mk_port();
    let ch0 = po0.mk_chan();
    ch.send(ch0.unsafe_ptr());
    let pch1;
    pch1 = po.recv();
    let ch1 = comm::chan_from_unsafe_ptr(pch1);
    // Does the transmitted channel still work?

    ch1.send(10);
    let i: int;
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
