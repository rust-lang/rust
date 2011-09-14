// -*- rust -*-

// error-pattern:1 == 2
use std;
import std::task;
import std::comm::port;
import std::comm::recv;

fn child() { assert (1 == 2); }

fn main() {
    let p = port::<int>();
    let f = child;
    task::spawn(f);
    let x = recv(p);
}
