use std;
import std::comm;
import std::task;
import std::uint;

fn child(c: comm::chan<~uint>, i: uint) {
    comm::send(c, ~i);
}

fn main() {
    let p = comm::port();
    let n = 100u;
    let expected = 0u;
    for each i in uint::range(0u, n) {
        let f = bind child(comm::chan(p), i);
        task::spawn(f);
        expected += i;
    }

    let actual = 0u;
    for each i in uint::range(0u, n) {
        let j = comm::recv(p);
        actual += *j;
    }

    assert expected == actual;
}