// -*- rust -*-
extern mod std;
use task::*;

fn main() {
    let mut result = None;
    task::task().future_result(|+r| { result = Some(move r); }).spawn(child);
    error!("1");
    yield();
    error!("2");
    yield();
    error!("3");
    future::get(&option::unwrap(move result));
}

fn child() {
    error!("4"); yield(); error!("5"); yield(); error!("6");
}
