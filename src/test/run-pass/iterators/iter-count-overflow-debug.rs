// run-pass
// only-32bit too impatient for 2⁶⁴ items
// ignore-wasm32-bare compiled with panic=abort by default
// compile-flags: -C debug_assertions=yes

use std::panic;
use std::usize::MAX;

fn main() {
    assert_eq!((0..MAX).by_ref().count(), MAX);

    let r = panic::catch_unwind(|| {
        (0..=MAX).by_ref().count()
    });
    assert!(r.is_err());
}
