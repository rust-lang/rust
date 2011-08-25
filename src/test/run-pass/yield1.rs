// -*- rust -*-
use std;
import std::task;
import std::task::*;

fn main() {
    let c = child;
    let other = task::spawn_joinable(c);
    log_err "1";
    yield();
    join(other);
}

fn child() { log_err "2"; }
