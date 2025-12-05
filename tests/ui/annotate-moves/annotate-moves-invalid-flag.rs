//@ check-fail
//@ compile-flags: -Z annotate-moves=invalid

// Test that invalid values for annotate-moves flag are rejected

fn main() {
    // This should fail at compile time due to invalid flag value
}

//~? ERROR incorrect value `invalid` for unstable option `annotate-moves`
