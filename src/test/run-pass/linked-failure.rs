// -*- rust -*-
// xfail-fast
// error-pattern:1 == 2
use std;
import std::task;
import std::comm::port;
import std::comm::recv;

fn child() { assert (1 == 2); }

fn parent() {
    // Since this task isn't supervised it won't bring down the whole
    // process
    task::unsupervise();
    let p = port::<int>();
    let f = child;
    task::spawn(f);
    let x = recv(p);
}

fn main() {
    let f = parent;
    task::spawn(f);
}