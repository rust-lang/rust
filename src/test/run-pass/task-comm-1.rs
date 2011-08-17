use std;

import std::task::spawn;
import std::task::join_id;

fn main() { test00(); }

fn start() { log "Started / Finished task."; }

fn test00() {
    let f = start;
    let t = spawn(f);
    join_id(t);
    log "Completing.";
}
