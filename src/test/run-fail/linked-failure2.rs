// -*- rust -*-

// error-pattern:fail
use std;
import std::task;
import std::comm::chan;
import std::comm::port;
import std::comm::recv;

fn child() { fail; }

fn main() {
    let p = port::<int>();
    let f = child;
    task::spawn(f);
    task::yield();
}
