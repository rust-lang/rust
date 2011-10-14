// xfail-win32 leaks
use std;
import std::task;
import std::comm;
import std::uint;

fn# die(&&_i: ()) {
    fail;
}

fn# iloop(&&_i: ()) {
    task::unsupervise();
    task::spawn((), die);
}

fn main() {
    for each i in uint::range(0u, 100u) {
        task::spawn((), iloop);
    }
}