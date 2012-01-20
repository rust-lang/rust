// -*- rust -*-

use std;
import comm;
import comm::send;
import comm::port;
import comm::recv;
import comm::chan;

// Tests of ports and channels on various types
fn test_rec() {
    type r = {val0: int, val1: u8, val2: char};

    let po = comm::port();
    let ch = chan(po);
    let r0: r = {val0: 0, val1: 1u8, val2: '2'};
    send(ch, r0);
    let r1: r;
    r1 = recv(po);
    assert (r1.val0 == 0);
    assert (r1.val1 == 1u8);
    assert (r1.val2 == '2');
}

fn test_vec() {
    let po = port();
    let ch = chan(po);
    let v0: [int] = [0, 1, 2];
    send(ch, v0);
    let v1 = recv(po);
    assert (v1[0] == 0);
    assert (v1[1] == 1);
    assert (v1[2] == 2);
}

fn test_str() {
    // FIXME: re-enable this once strings are unique and sendable
    /*
        let po = comm::mk_port();
        let ch = po.mk_chan();
        let s0: str = "test";
        send(ch, s0);
        let s1: str;
        s1 = po.recv();
        assert (s1.(0) as u8 == 't' as u8);
        assert (s1.(1) as u8 == 'e' as u8);
        assert (s1.(2) as u8 == 's' as u8);
        assert (s1.(3) as u8 == 't' as u8);
    */
}

fn test_tag() {
    enum t { tag1, tag2(int), tag3(int, u8, char), }
    let po = port();
    let ch = chan(po);
    send(ch, tag1);
    send(ch, tag2(10));
    send(ch, tag3(10, 11u8, 'A'));
    // FIXME: Do port semantics really guarantee these happen in order?
    let t1: t;
    t1 = recv(po);
    assert (t1 == tag1);
    t1 = recv(po);
    assert (t1 == tag2(10));
    t1 = recv(po);
    assert (t1 == tag3(10, 11u8, 'A'));
}

fn test_chan() {
    let po = port();
    let ch = chan(po);
    let po0 = port();
    let ch0 = chan(po0);
    send(ch, ch0);
    let ch1 = recv(po);
    // Does the transmitted channel still work?

    send(ch1, 10);
    let i: int;
    i = recv(po0);
    assert (i == 10);
}

fn main() { test_rec(); test_vec(); test_str(); test_tag(); test_chan(); }
