// -*- rust -*-
use std;
import task;
import task::*;

fn main() {
    let other = task::spawn_joinable((), child);
    #error("1");
    yield();
    join(other);
}

fn child(&&_i: ()) { #error("2"); }
