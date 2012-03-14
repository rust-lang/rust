// -*- rust -*-
use std;
import task;
import task::*;

fn main() {
    let builder = task::task_builder();
    let result = task::future_result(builder);
    task::run(builder) {|| child(); }
    #error("1");
    yield();
    future::get(result);
}

fn child() { #error("2"); }
