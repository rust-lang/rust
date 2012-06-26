

// -*- rust -*-
use std;

fn grow(&v: [int]/~) { v += [1]/~; }

fn main() {
    let mut v: [int]/~ = []/~;
    grow(v);
    grow(v);
    grow(v);
    let len = vec::len::<int>(v);
    log(debug, len);
    assert (len == 3 as uint);
}
