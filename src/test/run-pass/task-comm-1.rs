use std;

import std::task::spawn_joinable2;
import std::task::join;

fn main() { test00(); }

fn# start(&&_i: ()) { log "Started / Finished task."; }

fn test00() {
    let t = spawn_joinable2((), start);
    join(t);
    log "Completing.";
}
