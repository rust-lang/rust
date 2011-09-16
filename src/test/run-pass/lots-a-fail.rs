// FIXME: Importing std::task doesn't work under check-fast?!
// xfail-fast
use std;
import std::task;
import std::comm;
import std::uint;

fn die() {
    fail;
}

fn iloop() {
    task::unsupervise();
    let f = die;
    task::spawn(f);
}

fn main() {
    for each i in uint::range(0u, 100u) {
        let f = iloop;
        task::spawn(f);
    }
}