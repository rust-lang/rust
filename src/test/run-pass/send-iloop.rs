// xfail-win32
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
    let p = comm::port::<()>();
    let c = comm::chan(p);
    while true {
        comm::send(c, ());
    }
}

fn main() {
    for each i in uint::range(0u, 16u) {
        let f = iloop;
        task::spawn(f);
    }
}