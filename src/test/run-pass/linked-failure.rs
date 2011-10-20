// -*- rust -*-
// xfail-win32
use std;
import std::task;
import std::comm::port;
import std::comm::recv;

fn# child(&&_i: ()) { assert (1 == 2); }

fn# parent(&&_i: ()) {
    // Since this task isn't supervised it won't bring down the whole
    // process
    task::unsupervise();
    let p = port::<int>();
    task::spawn((), child);
    let x = recv(p);
}

fn main() {
    task::spawn((), parent);
}