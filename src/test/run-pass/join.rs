// -*- rust -*-

use std;

import std::task::*;

fn main() {
    let f = child;
    let other = _spawn(f);
    log_err "1";
    yield();
    join_id(other);
    log_err "3";
}

fn child() { log_err "2"; }