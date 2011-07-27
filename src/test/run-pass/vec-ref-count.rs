
use std;
import std::vec;

fn main() {
    let v = [1, 2, 3];
    log_err vec::refcount[int](v);
    log_err vec::refcount[int](v);
    log_err vec::refcount[int](v);
    assert (vec::refcount[int](v) == 1u || vec::refcount[int](v) == 2u);
    assert (vec::refcount[int](v) == 1u || vec::refcount[int](v) == 2u);
}