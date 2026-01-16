// Library crate for testing coverage instrumentation with -Z separate-spans.
// Contains functions with various control flow for coverage testing.

#![crate_type = "rlib"]
#![crate_name = "rdr_coverage_lib"]

/// Simple function with linear control flow.
pub fn simple_add(a: i32, b: i32) -> i32 {
    a + b
}

/// Function with branching for coverage testing.
pub fn max_value(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

/// Function with a loop for coverage testing.
pub fn sum_to_n(n: i32) -> i32 {
    let mut sum = 0;
    for i in 1..=n {
        sum += i;
    }
    sum
}

/// Function with multiple branches.
pub fn classify_number(n: i32) -> &'static str {
    if n < 0 {
        "negative"
    } else if n == 0 {
        "zero"
    } else if n < 10 {
        "small positive"
    } else {
        "large positive"
    }
}

// Private function for testing coverage across visibility boundaries.
fn private_multiply(a: i32, b: i32) -> i32 {
    a * b
}

/// Public function that uses private helper.
pub fn square(n: i32) -> i32 {
    private_multiply(n, n)
}
