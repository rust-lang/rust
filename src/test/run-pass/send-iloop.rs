// xfail-win32
use std;
import std::task;
import std::comm;
import std::uint;

fn die(&&_i: ()) {
    fail;
}

fn iloop(&&_i: ()) {
    task::unsupervise();
    task::spawn((), die);
    let p = comm::port::<()>();
    let c = comm::chan(p);
    while true {
        comm::send(c, ());
    }
}

fn main() {
    uint::range(0u, 16u) {|_i|
        task::spawn((), iloop);
    }
}