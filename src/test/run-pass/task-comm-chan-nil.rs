// -*- rust -*-

use std;
import std::comm;

// rustboot can't transmit nils across channels because they don't have
// any size, but rustc currently can because they do have size. Whether
// or not this is desirable I don't know, but here's a regression test.
fn main() {
    let po: comm::_port[()] = comm::mk_port();
    let ch: comm::_chan[()] = po.mk_chan();
    comm::send(ch, ());
    let n: () = po.recv();
    assert (n == ());
}