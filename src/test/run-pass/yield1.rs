// -*- rust -*-
use std;
use task::*;

fn main() {
    let mut result = None;
    task::task().future_result(|+r| { result = Some(r); }).spawn(child);
    error!("1");
    yield();
    future::get(&option::unwrap(result));
}

fn child() { error!("2"); }
