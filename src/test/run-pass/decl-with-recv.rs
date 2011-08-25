// -*- rust -*-

use std;
import std::comm::port;
import std::comm::chan;
import std::comm::send;
import std::comm::recv;

fn main() {
    let po = port();
    let ch = chan(po);
    send(ch, 10);
    let i = recv(po);
    assert (i == 10);
    send(ch, 11);
    let j = recv(po);
    assert (j == 11);
}
