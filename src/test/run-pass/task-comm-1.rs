use std;

import std::task::spawn_joinable;
import std::task::join;

fn main() { test00(); }

fn start(&&_i: ()) { log "Started / Finished task."; }

fn test00() {
    let t = spawn_joinable((), start);
    join(t);
    log "Completing.";
}
