//@ check-fail
//@ compile-flags: -Z instrument-moves=invalid

// Test that invalid values for instrument-moves flag are rejected

fn main() {
    // This should fail at compile time due to invalid flag value
}

//~? ERROR incorrect value `invalid` for unstable option `instrument-moves`
