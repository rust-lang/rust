use std;
import std::task;
import std::comm;

fn main() { test00(); }

fn test00_start(pc: *u8, number_of_messages: int) {
    let c = comm::chan_from_unsafe_ptr(pc);
    let i: int = 0;
    while i < number_of_messages { c.send(i); i += 1; }
}

fn test00() {
    let r: int = 0;
    let sum: int = 0;
    let p = comm::mk_port();
    let number_of_messages: int = 10;

    let t0: task = spawn test00_start(p.mk_chan().unsafe_ptr(),
                                      number_of_messages);

    let i: int = 0;
    while i < number_of_messages { r = p.recv(); sum += r; log r; i += 1; }

    task::join(t0);

    assert (sum == number_of_messages * (number_of_messages - 1) / 2);
}