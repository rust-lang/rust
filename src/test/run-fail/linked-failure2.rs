// -*- rust -*-

// error-pattern:fail
use std;
import std::task;
import std::comm::chan;
import std::comm::port;
import std::comm::recv;

fn child(&&_i: ()) { fail; }

fn main() {
    let p = port::<int>();
    task::spawn((), child);
    task::yield();
}
