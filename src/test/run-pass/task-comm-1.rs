use std;

import std::task::join;

fn main() { test00(); }

fn start() { log "Started / Finished task."; }

fn test00() { let t: task = spawn start(); join(t); log "Completing."; }
