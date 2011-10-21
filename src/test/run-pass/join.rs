// -*- rust -*-

use std;

import std::task::*;

fn main() {
    let other = spawn_joinable((), child);
    log_err "1";
    yield();
    join(other);
    log_err "3";
}

fn child(&&_i: ()) { log_err "2"; }
