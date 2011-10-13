// -*- rust -*-
use std;
import std::task;
import std::task::*;

fn main() {
    let other = task::spawn_joinable2((), child);
    log_err "1";
    yield();
    join(other);
}

fn# child(&&_i: ()) { log_err "2"; }
