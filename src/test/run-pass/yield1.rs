// -*- rust -*-
use std;
import task;
import task::*;

fn main() {
    let mut result = none;
    task::task().future_result(|+r| { result = some(r); }).spawn(child);
    error!{"1"};
    yield();
    future::get(&option::unwrap(result));
}

fn child() { error!{"2"}; }
