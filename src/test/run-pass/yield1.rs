// -*- rust -*-
use std;
import std::task;
import std::task::*;

fn main() {
    let other = task::_spawn(bind child());
    log_err "1"; yield();
    join_id(other);
}

fn child() { log_err "2"; }
