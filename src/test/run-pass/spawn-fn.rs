// -*- rust -*-

use std;
import std::task::yield;
import std::task;

fn x(s: str, n: int) { log s; log n; }

fn main() {
    task::spawn(bind x("hello from first spawned fn", 65));
    task::spawn(bind x("hello from second spawned fn", 66));
    task::spawn(bind x("hello from third spawned fn", 67));
    let i: int = 30;
    while i > 0 { i = i - 1; log "parent sleeping"; yield(); }
}
