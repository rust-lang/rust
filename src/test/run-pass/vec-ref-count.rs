use std;
import std::_vec;

fn main() {
    auto v = vec(1, 2, 3);
    log_err _vec::refcount[int](v);
    log_err _vec::refcount[int](v);
    log_err _vec::refcount[int](v);
    assert (_vec::refcount[int](v) == 1u || _vec::refcount[int](v) == 2u);
    assert (_vec::refcount[int](v) == 1u || _vec::refcount[int](v) == 2u);
}

