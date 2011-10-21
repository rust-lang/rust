// -*- rust -*-

// error-pattern:fail
use std;
import std::task;
import std::comm::port;
import std::comm::recv;

fn grandchild(&&_i: ()) { fail; }

fn child(&&_i: ()) {
    let p = port::<int>();
    task::spawn((), grandchild);
    let x = recv(p);
}

fn main() {
    let p = port::<int>();
    task::spawn((), child);
    let x = recv(p);
}
