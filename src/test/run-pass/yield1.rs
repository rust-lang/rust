// -*- rust -*-
use std;
import task;
import task::*;

fn main() {
    let other = task::spawn_joinable((), child);
    log_err "1";
    yield();
    join(other);
}

fn child(&&_i: ()) { log_err "2"; }
