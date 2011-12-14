
use std;
import vec::*;

fn main() {
    let v = init_elt(0, 0u);
    v += [4, 2];
    assert (reversed(v) == [2, 4]);
}
