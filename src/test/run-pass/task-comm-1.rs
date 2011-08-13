use std;

import std::task::_spawn;
import std::task::join_id;

fn main() { test00(); }

fn start() { log "Started / Finished task."; }

fn test00() { let t = _spawn(bind start()); join_id(t); log "Completing."; }
