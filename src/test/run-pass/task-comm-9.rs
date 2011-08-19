use std;
import std::task;
import std::comm;

fn main() { test00(); }

fn test00_start(c: comm::_chan<int>, number_of_messages: int) {
    let i: int = 0;
    while i < number_of_messages { comm::send(c, i + 0); i += 1; }
}

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p = comm::mk_port();
    let number_of_messages: int = 10;

    let t0 = task::_spawn(bind test00_start(p.mk_chan(), number_of_messages));

    let i: int = 0;
    while i < number_of_messages { r = p.recv(); sum += r; log r; i += 1; }

    task::join_id(t0);

    assert (sum == number_of_messages * (number_of_messages - 1) / 2);
}
