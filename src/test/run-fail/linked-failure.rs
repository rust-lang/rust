// -*- rust -*-

// error-pattern:1 == 2
// no-valgrind

use std;
import std::task;
import std::comm::mk_port;

fn child() { assert (1 == 2); }

fn main() {
    let p = mk_port::<int>();
    let f = child;
    task::_spawn(f);
    let x = p.recv();
}
