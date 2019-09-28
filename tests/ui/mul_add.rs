#![warn(clippy::manual_mul_add)]
#![allow(unused_variables)]

fn mul_add_test() {
    let a: f64 = 1234.567;
    let b: f64 = 45.67834;
    let c: f64 = 0.0004;

    // Examples of not auto-fixable expressions
    let test1 = (a * b + c) * (c + a * b) + (c + (a * b) + c);
    let test2 = 1234.567 * 45.67834 + 0.0004;
}

fn main() {
    mul_add_test();
}
