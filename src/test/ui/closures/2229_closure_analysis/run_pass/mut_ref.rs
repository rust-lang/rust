// run-pass

// Test that we can mutate a capture that
// contains a dereferenced mutable reference from within the closure

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete

fn main() {
    let mut x = String::new();
    let rx = &mut x;

    let mut c = || {
        *rx = String::new();
    };

    c();
}
