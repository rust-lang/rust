// Main crate that exercises the library functions for coverage.

extern crate rdr_coverage_lib;

use rdr_coverage_lib::*;

fn main() {
    // Exercise all functions to generate coverage data

    // Simple add
    let _ = simple_add(1, 2);

    // Max value - exercise both branches
    let _ = max_value(5, 3); // a > b branch
    let _ = max_value(2, 7); // else branch

    // Sum to n
    let _ = sum_to_n(10);

    // Classify number - exercise all branches
    let _ = classify_number(-5); // negative
    let _ = classify_number(0); // zero
    let _ = classify_number(5); // small positive
    let _ = classify_number(100); // large positive

    // Square (calls private helper)
    let _ = square(4);

    println!("Coverage test completed");
}
