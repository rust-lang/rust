use std;

import std::task;
import std::comm;
import std::comm::_chan;
import std::comm::_port;
import std::comm::mk_port;
import std::comm::send;

fn producer(c: _chan[[u8]]) {
    send(c, [1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8,
             8u8, 9u8, 10u8, 11u8, 12u8, 13u8 ]);
}

fn main() {
    let p: _port[[u8]] = mk_port();
    let prod = task::_spawn(bind producer(p.mk_chan()));

    let data: [u8] = p.recv();
}
