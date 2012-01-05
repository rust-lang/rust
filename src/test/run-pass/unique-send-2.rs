use std;
import comm;
import task;
import uint;

fn child(c: comm::chan<~uint>, i: uint) {
    comm::send(c, ~i);
}

fn main() {
    let p = comm::port();
    let ch = comm::chan(p);
    let n = 100u;
    let expected = 0u;
    uint::range(0u, n) {|i|
        task::spawn {|| child(ch, i); };
        expected += i;
    }

    let actual = 0u;
    uint::range(0u, n) {|_i|
        let j = comm::recv(p);
        actual += *j;
    }

    assert expected == actual;
}