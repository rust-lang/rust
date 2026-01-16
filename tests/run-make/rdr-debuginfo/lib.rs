// Library crate for testing debuginfo with -Z separate-spans.
// Contains functions at known line numbers for DWARF verification.

#![crate_type = "rlib"]
#![crate_name = "rdr_debuginfo_lib"]

/// Public function at a known line for debuginfo testing.
/// Should appear at line 9 in DWARF info.
pub fn public_function(x: i32) -> i32 {
    // Line 9
    let y = x + 1; // Line 10
    let z = y * 2; // Line 11
    z // Line 12
}

/// Another public function with more complex control flow.
/// Tests that debuginfo correctly handles multiple statements.
pub fn complex_function(a: i32, b: i32) -> i32 {
    // Line 18
    let mut result = 0; // Line 19
    if a > b {
        // Line 20
        result = a - b; // Line 21
    } else {
        // Line 22
        result = b - a; // Line 23
    } // Line 24
    result // Line 25
}

// Private function - its line info should still be correct
// even though it's not part of the public API.
fn private_helper(x: i32) -> i32 {
    // Line 30
    x * x // Line 31
}

/// Public function that calls a private helper.
/// Tests that debuginfo works across visibility boundaries.
pub fn calls_private(x: i32) -> i32 {
    // Line 36
    private_helper(x) + 1 // Line 37
}
