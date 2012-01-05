use std;

import task::spawn_joinable;
import task::join;

fn main() { test00(); }

fn start() { #debug("Started / Finished task."); }

fn test00() {
    let t = spawn_joinable {|| start(); };
    join(t);
    #debug("Completing.");
}
