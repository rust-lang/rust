// xfail-fast
// aux-build:packed.rs

extern mod packed;

use std::mem;

fn main() {
    assert_eq!(mem::size_of::<packed::S>(), 5);
}
