// -*- rust -*-
use std;
import std::task::*;

fn main() {
    let other = spawn child();
    log_err "1";
    yield();
    log_err "2";
    yield();
    log_err "3";
    join(other);
}

fn child() { log_err "4"; yield(); log_err "5"; yield(); log_err "6"; }