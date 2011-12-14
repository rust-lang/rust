// -*- rust -*-

use std;
import task::yield;
import task;

fn x(&&args: (str, int)) {
    let (s, n) = args;
    log s; log n;
}

fn main() {
    task::spawn(("hello from first spawned fn", 65), x);
    task::spawn(("hello from second spawned fn", 66), x);
    task::spawn(("hello from third spawned fn", 67), x);
    let i: int = 30;
    while i > 0 { i = i - 1; log "parent sleeping"; yield(); }
}
