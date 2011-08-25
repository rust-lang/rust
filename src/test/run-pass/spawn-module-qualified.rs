use std;
import std::task::join;
import std::task::spawn_joinable;

fn main() {
    let x = spawn_joinable(bind m::child(10));
    join(x);
}

mod m {
    fn child(i: int) { log i; }
}
