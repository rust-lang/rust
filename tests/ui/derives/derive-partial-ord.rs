// Checks that in a derived implementation of PartialOrd the lt, le, ge, gt methods are consistent
// with partial_cmp. Also verifies that implementation is consistent with that for tuples.
//
//@ run-pass

#[derive(PartialEq, PartialOrd)]
struct P(f64, f64);

fn main() {
    let values: &[f64] = &[1.0, 2.0, f64::NAN];
    for a in values {
        for b in values {
            for c in values {
                for d in values {
                    // Check impl for a tuple.
                    check(&(*a, *b), &(*c, *d));

                    // Check derived impl.
                    check(&P(*a, *b), &P(*c, *d));

                    // Check that impls agree with each other.
                    assert_eq!(
                        PartialOrd::partial_cmp(&(*a, *b), &(*c, *d)),
                        PartialOrd::partial_cmp(&P(*a, *b), &P(*c, *d)),
                    );
                }
            }
        }
    }
}

fn check<T: PartialOrd>(a: &T, b: &T) {
    use std::cmp::Ordering::*;
    match PartialOrd::partial_cmp(a, b) {
        None => {
            assert!(!(a < b));
            assert!(!(a <= b));
            assert!(!(a > b));
            assert!(!(a >= b));
        }
        Some(Equal) => {
            assert!(!(a < b));
            assert!(a <= b);
            assert!(!(a > b));
            assert!(a >= b);
        }
        Some(Less) => {
            assert!(a < b);
            assert!(a <= b);
            assert!(!(a > b));
            assert!(!(a >= b));
        }
        Some(Greater) => {
            assert!(!(a < b));
            assert!(!(a <= b));
            assert!(a > b);
            assert!(a >= b);
        }
    }
}
