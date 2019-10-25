// run-pass
// only-32bit too impatient for 2⁶⁴ items
// ignore-emscripten compiled with panic=abort by default
// compile-flags: -C debug_assertions=yes -C opt-level=3

use std::panic;
use std::usize::MAX;

fn main() {
    let n = MAX as u64;
    assert_eq!((0..).by_ref().position(|i| i >= n), Some(MAX));

    let r = panic::catch_unwind(|| {
        (0..).by_ref().position(|i| i > n)
    });
    assert!(r.is_err());

    let r = panic::catch_unwind(|| {
        (0..=n + 1).by_ref().position(|_| false)
    });
    assert!(r.is_err());
}
