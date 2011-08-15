

// -*- rust -*-
use std;

fn grow(v: &mutable [int]) { v += ~[1]; }

fn main() {
    let v: [int] = ~[];
    grow(v);
    grow(v);
    grow(v);
    let len = std::vec::len[int](v);
    log len;
    assert (len == 3 as uint);
}