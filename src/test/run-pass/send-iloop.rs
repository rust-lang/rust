// xfail-win32
use std;
import std::task;
import std::comm;
import std::uint;

fn# die(&&_i: ()) {
    fail;
}

fn# iloop(&&_i: ()) {
    task::unsupervise();
    task::spawn2((), die);
    let p = comm::port::<()>();
    let c = comm::chan(p);
    while true {
        comm::send(c, ());
    }
}

fn main() {
    for each i in uint::range(0u, 16u) {
        task::spawn2((), iloop);
    }
}