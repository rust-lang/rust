// run-pass

#![feature(destructuring_assignment)]

#![warn(unused_assignments)]

fn main() {
    let mut a;
    // Assignment occurs left-to-right.
    // However, we emit warnings when this happens, so it is clear that this is happening.
    (a, a) = (0, 1); //~ WARN value assigned to `a` is never read
    assert_eq!(a, 1);
}
