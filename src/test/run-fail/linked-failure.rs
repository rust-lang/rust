// -*- rust -*-

// error-pattern:1 == 2
// no-valgrind

use std;
import std::task;

fn child() { assert (1 == 2); }

fn main() {
    let p: port[int] = port();
    task::_spawn(bind child());
    let x: int; p |> x;
}
