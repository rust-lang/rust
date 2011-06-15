

// -*- rust -*-
use std;

fn grow(&mutable vec[int] v) { v += [1]; }

fn main() {
    let vec[int] v = [];
    grow(v);
    grow(v);
    grow(v);
    auto len = std::vec::len[int](v);
    log len;
    assert (len == 3 as uint);
}