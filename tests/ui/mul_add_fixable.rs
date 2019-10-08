// run-rustfix

#![warn(clippy::manual_mul_add)]
#![allow(unused_variables)]

fn mul_add_test() {
    let a: f64 = 1234.567;
    let b: f64 = 45.67834;
    let c: f64 = 0.0004;

    // Auto-fixable examples
    let test1 = a * b + c;
    let test2 = c + a * b;

    let test3 = (a * b) + c;
    let test4 = c + (a * b);

    let test5 = a.mul_add(b, c) * a.mul_add(b, c) + a.mul_add(b, c) + c;
    let test6 = 1234.567_f64 * 45.67834_f64 + 0.0004_f64;
}

fn main() {
    mul_add_test();
}
