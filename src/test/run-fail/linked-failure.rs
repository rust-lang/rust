// -*- rust -*-

// error-pattern:1 == 2
use std;
import std::task;
import std::comm::port;
import std::comm::recv;

fn# child(&&_i: ()) { assert (1 == 2); }

fn main() {
    let p = port::<int>();
    task::spawn2((), child);
    let x = recv(p);
}
