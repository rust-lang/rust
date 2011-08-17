// -*- rust -*-
use std;
import std::task;
import std::task::*;

fn main() {
    let c = child;
    let other = task::spawn(c);
    log_err "1"; yield();
    join_id(other);
}

fn child() { log_err "2"; }
