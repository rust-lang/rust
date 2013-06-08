// xfail-fast
// aux-build:packed.rs

extern mod packed;

use std::sys;

pub fn main() {
    assert_eq!(sys::size_of::<packed::S>(), 5);
}
