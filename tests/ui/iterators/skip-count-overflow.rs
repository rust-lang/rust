//@ run-pass
//@ only-32bit too impatient for 2⁶⁴ items
//@ compile-flags: -C overflow-checks -C opt-level=3

fn main() {
    let i = (0..usize::MAX).chain(0..10).skip(usize::MAX);
    assert_eq!(i.count(), 10);
}
