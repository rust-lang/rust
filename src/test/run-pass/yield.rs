// -*- rust -*-
use std;
import task;
import task::*;

fn main() {
    let other = task::spawn_joinable {|| child(); };
    #error("1");
    yield();
    #error("2");
    yield();
    #error("3");
    join(other);
}

fn child() {
    #error("4"); yield(); #error("5"); yield(); #error("6");
}
