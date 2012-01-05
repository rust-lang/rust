// -*- rust -*-

use std;
import task;

fn main() {
    let i = 10;
    while i > 0 { task::spawn {|| child(i); }; i = i - 1; }
    #debug("main thread exiting");
}

fn child(&&x: int) { log(debug, x); }

