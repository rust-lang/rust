// -*- rust -*-
use std;
import task;
import task::*;

fn main() {
    let other = task::spawn_joinable((), child);
    log_err "1";
    yield();
    log_err "2";
    yield();
    log_err "3";
    join(other);
}

fn child(&&_i: ()) {
    log_err "4"; yield(); log_err "5"; yield(); log_err "6";
}
