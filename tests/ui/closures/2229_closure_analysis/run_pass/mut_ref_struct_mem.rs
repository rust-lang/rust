//@ edition:2021
//@ run-pass

// Test that we can mutate a place through a mut-borrow
// that is captured by the closure

// More specifically we test that the if the mutable reference isn't root variable of a capture
// but rather accessed while accessing the precise capture.

fn mut_tuple() {
    let mut t = (10, 10);

    let t1 = (&mut t, 10);

    let mut c = || {
        // Mutable because (*t.0) is mutable
        t1.0.0 += 10;
    };

    c();
}

fn mut_tuple_nested() {
    let mut t = (10, 10);

    let t1 = (&mut t, 10);

    let mut c = || {
        let mut c = || {
            // Mutable because (*t.0) is mutable
            t1.0.0 += 10;
        };

        c();
    };

    c();
}

fn main() {
    mut_tuple();
    mut_tuple_nested();
}
