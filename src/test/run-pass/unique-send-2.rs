use std;
import std::comm;
import std::task;
import std::uint;

fn child(args: (comm::chan<~uint>, uint)) {
    let (c, i) = args;
    comm::send(c, ~i);
}

fn main() {
    let p = comm::port();
    let n = 100u;
    let expected = 0u;
    uint::range(0u, n) {|i|
        task::spawn((comm::chan(p), i), child);
        expected += i;
    }

    let actual = 0u;
    uint::range(0u, n) {|_i|
        let j = comm::recv(p);
        actual += *j;
    }

    assert expected == actual;
}