// -*- rust -*-

// error-pattern:fail
use std;
import std::task;
import std::comm::port;
import std::comm::recv;

fn grandchild() { fail; }

fn child() {
    let p = port::<int>();
    let f = grandchild;
    task::spawn(f);
    let x = recv(p);
}

fn main() {
    let p = port::<int>();
    let f = child;
    task::spawn(f);
    let x = recv(p);
}
