// -*- rust -*-

use std;
import std::task::yield;
import std::task;

fn# x(&&args: (str, int)) {
    let (s, n) = args;
    log s; log n;
}

fn main() {
    task::spawn2(("hello from first spawned fn", 65), x);
    task::spawn2(("hello from second spawned fn", 66), x);
    task::spawn2(("hello from third spawned fn", 67), x);
    let i: int = 30;
    while i > 0 { i = i - 1; log "parent sleeping"; yield(); }
}
