// xfail-stage0

use std;
import std._vec;

fn main() {
    auto v = vec(1, 2, 3);
    check (_vec.refcount[int](v) == 1u);
    check (_vec.refcount[int](v) == 1u);
}

