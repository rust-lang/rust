// -*- rust -*-

use std;
import std::comm::mk_port;
import std::comm::send;

fn main() {
    let po = mk_port();
    let ch = po.mk_chan();
    send(ch, 10);
    let i = po.recv();
    assert (i == 10);
    send(ch, 11);
    let j = po.recv();
    assert (j == 11);
}
