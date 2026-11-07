//@ check-fail
//@ compile-flags: -Z annotate-moves=-5

// Test that negative size limits are rejected

fn main() {
    // This should fail at compile time due to invalid negative size limit
}

//~? ERROR incorrect value `-5` for unstable option `annotate-moves`
