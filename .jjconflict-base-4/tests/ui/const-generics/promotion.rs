//@ run-pass
// tests that promoting expressions containing const parameters is allowed.
fn promotion_test<const N: usize>() -> &'static usize {
    &(3 + N)
}

fn main() {
    assert_eq!(promotion_test::<13>(), &16);
}
