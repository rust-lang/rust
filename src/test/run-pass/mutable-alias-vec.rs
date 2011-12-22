

// -*- rust -*-
use std;

fn grow(&v: [int]) { v += [1]; }

fn main() {
    let v: [int] = [];
    grow(v);
    grow(v);
    grow(v);
    let len = vec::len::<int>(v);
    log_full(core::debug, len);
    assert (len == 3 as uint);
}
