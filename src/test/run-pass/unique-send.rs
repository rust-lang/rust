use std;
import std::comm;
import std::task;

fn main() {
    let p = comm::port();
    let c = comm::chan(p);
    comm::send(c, ~100);
    let v = comm::recv(p);
    assert v == ~100;
}