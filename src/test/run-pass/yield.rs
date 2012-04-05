// -*- rust -*-
use std;
import task;
import task::*;

fn main() {
    let builder = task::builder();
    let result = task::future_result(builder);
    task::run(builder) {|| child(); }
    #error("1");
    yield();
    #error("2");
    yield();
    #error("3");
    future::get(result);
}

fn child() {
    #error("4"); yield(); #error("5"); yield(); #error("6");
}
