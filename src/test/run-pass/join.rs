// -*- rust -*-

use std;

import std::task::*;

fn main() {
    let other = _spawn(bind child());
    log_err "1";
    yield();
    join_id(other);
    log_err "3";
}

fn child() { log_err "2"; }