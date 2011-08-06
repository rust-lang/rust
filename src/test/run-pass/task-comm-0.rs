use std;

import std::comm;

fn main() { test05(); }

fn test05_start(pch: *u8) {
    let ch = comm::chan_from_unsafe_ptr(pch);

    ch.send(10);
    ch.send(20);
    ch.send(30);
}

fn test05() {
    let po = comm::mk_port[int]();
    let ch = po.mk_chan();
    spawn test05_start(ch.unsafe_ptr());
    let value = po.recv();
    value = po.recv();
    value = po.recv();
    assert (value == 30);
}