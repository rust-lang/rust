// -*- rust -*-

use std;

import task::*;

fn main() {
    let other = spawn_joinable {|| child(); };
    #error("1");
    yield();
    join(other);
    #error("3");
}

fn child() { #error("2"); }
