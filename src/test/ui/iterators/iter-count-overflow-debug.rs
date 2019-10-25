// run-pass
// only-32bit too impatient for 2⁶⁴ items
// ignore-emscripten compiled with panic=abort by default
// compile-flags: -C debug_assertions=yes -C opt-level=3

use std::panic;
use std::usize::MAX;

fn main() {
    assert_eq!((0..MAX).by_ref().count(), MAX);

    let r = panic::catch_unwind(|| {
        (0..=MAX).by_ref().count()
    });
    assert!(r.is_err());
}
