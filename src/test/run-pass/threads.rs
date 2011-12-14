// -*- rust -*-

use std;
import task;

fn main() {
    let i = 10;
    while i > 0 { task::spawn(copy i, child); i = i - 1; }
    log "main thread exiting";
}

fn child(&&x: int) { log x; }

