// Regression test for issue #91725.
//
// run-pass
// compile-flags: -Zmir-opt-level=4

fn main() {
    let a = true;
    let _ = &a;
    let mut b = false;
    b |= a;
    assert!(b);
}
