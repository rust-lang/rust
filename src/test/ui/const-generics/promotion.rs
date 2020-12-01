// run-pass
// tests that promoting expressions containing const parameters is allowed.
#![feature(min_const_generics)]

fn promotion_test<const N: usize>() -> &'static usize {
    &(3 + N)
}

fn main() {
    assert_eq!(promotion_test::<13>(), &16);
}
