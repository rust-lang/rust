
use std;
import vec::*;

fn main() {
    let mut v = from_elem(0u, 0);
    v += [4, 2];
    assert (reversed(v) == [2, 4]);
}
