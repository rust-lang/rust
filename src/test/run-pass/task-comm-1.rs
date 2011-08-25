use std;

import std::task::spawn_joinable;
import std::task::join;

fn main() { test00(); }

fn start() { log "Started / Finished task."; }

fn test00() {
    let f = start;
    let t = spawn_joinable(f);
    join(t);
    log "Completing.";
}
