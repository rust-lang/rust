//! Test early return within binary operation expressions

//@ run-pass

#![allow(dead_code)]
#![allow(unreachable_code)]

use std::ops::Add;

/// Function that performs addition with an early return in the right operand
fn add_with_early_return<T: Add<Output = T> + Copy>(n: T) -> T {
    n + { return n }
}

pub fn main() {
    // Test with different numeric types to ensure generic behavior works
    let _result1 = add_with_early_return(42i32);
    let _result2 = add_with_early_return(3.14f64);
}
