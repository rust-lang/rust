//xfail-stage0
//xfail-test

use std;
import std::task;

fn f(x : @{a:int, b:int}) {
    assert (x.a == 10);
    assert (x.b == 12);
}

fn main() {
    let z : @{a:int, b:int} = @{ a : 10, b : 12};
    let p = task::_spawn(bind f(z));
    task::join_id(p);
}