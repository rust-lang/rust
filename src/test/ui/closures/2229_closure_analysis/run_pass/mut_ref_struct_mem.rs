// run-pass

// Test that we can mutate a place through a mut-borrow
// that is captured by the closure
//
// More specifically we test that the if the mutable reference isn't root variable of a capture
// but rather accessed while acessing the precise capture.

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete

fn main() {
    let mut t = (10, 10);

    let t1 = (&mut t, 10);

    let mut c = || {
        // Mutable because (*t.0) is mutable
        t1.0.0 += 10;
    };

    c();
}
