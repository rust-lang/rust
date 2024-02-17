//@ run-pass
//@ only-32bit too impatient for 2⁶⁴ items
//@ compile-flags: -C debug_assertions=no -C opt-level=3

fn main() {
    assert_eq!((0..usize::MAX).by_ref().count(), usize::MAX);
    assert_eq!((0..=usize::MAX).by_ref().count(), 0);
}
