// error-pattern:meep
use std;
import std::comm::chan;
import std::comm::port;
import std::comm::send;
import std::comm::recv;

fn echo<~T>(c: chan<T>, oc: chan<chan<T>>) {
    // Tests that the type argument in port gets
    // visited
    let p = port::<T>();
    send(oc, chan(p));

    let x = recv(p);
    send(c, x);
}

fn main() { fail "meep"; }
