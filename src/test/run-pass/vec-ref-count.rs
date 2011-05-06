use std;
import std.Vec;

fn main() {
    auto v = vec(1, 2, 3);
    log_err Vec.refcount[int](v);
    log_err Vec.refcount[int](v);
    log_err Vec.refcount[int](v);
    assert (Vec.refcount[int](v) == 1u || Vec.refcount[int](v) == 2u);
    assert (Vec.refcount[int](v) == 1u || Vec.refcount[int](v) == 2u);
}

