// error-pattern:meep
use std;
import std::comm::_chan;
import std::comm::mk_port;
import std::comm::send;

fn echo<~T>(c: _chan<T>, oc: _chan<_chan<T>>) {
    // Tests that the type argument in port gets
    // visited
    let p = mk_port[T]();
    send(oc, p.mk_chan());

    let x = p.recv();
    send(c, x);
}

fn main() { fail "meep"; }
