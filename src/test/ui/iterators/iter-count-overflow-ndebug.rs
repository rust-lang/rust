// run-pass
// only-32bit too impatient for 2⁶⁴ items
// compile-flags: -C debug_assertions=no -C opt-level=3

use std::usize::MAX;

fn main() {
    assert_eq!((0..MAX).by_ref().count(), MAX);
    assert_eq!((0..=MAX).by_ref().count(), 0);
}
