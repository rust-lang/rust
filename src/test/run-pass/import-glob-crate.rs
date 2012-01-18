
use std;
import vec::*;

fn main() {
    let v = init_elt(0u, 0);
    v += [4, 2];
    assert (reversed(v) == [2, 4]);
}
