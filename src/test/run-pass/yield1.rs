

// xfail-stage0
// -*- rust -*-
use std;
import std::task::*;

fn main() { let other = spawn child(); log_err "1"; yield(); join(other); }

fn child() { log_err "2"; }