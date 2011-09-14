// -*- rust -*-
// xfail-test
// error-pattern:1 == 2
use std;
import std::task;
import std::comm::chan;
import std::comm::port;
import std::comm::recv;

fn child() { assert (1 == 2); }

fn parent() {
    let p = port::<int>();
    let f = child;
    task::spawn(f);
    let x = recv(p);
}

// This task is not linked to the failure chain, but since the other
// tasks are going to fail the kernel, this one will fail too
fn sleeper() {
    let p = port::<int>();
    let x = recv(p);
}

fn main() {
    let f = parent;
    let g = sleeper;
    task::spawn(f);
    task::spawn(g);
}