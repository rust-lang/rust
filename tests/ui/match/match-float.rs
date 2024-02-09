// run-pass
// Makes sure we use `==` (not bitwise) semantics for float comparison.

fn main() {
    const F1: f32 = 0.0;
    const F2: f32 = -0.0;
    assert_eq!(F1, F2);
    assert_ne!(F1.to_bits(), F2.to_bits());
    assert!(matches!(F1, F2));
    assert!(matches!(F2, F1));
}
