// -*- rust -*-
extern mod std;
use task::*;

fn main() {
    let mut result = None;
    task::task().future_result(|+r| { result = Some(move r); }).spawn(child);
    error!("1");
    yield();
    option::unwrap(move result).recv();
}

fn child() { error!("2"); }
