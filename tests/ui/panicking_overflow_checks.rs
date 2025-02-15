#![warn(clippy::panicking_overflow_checks)]
#![allow(clippy::needless_if)]

fn test(a: u32, b: u32, c: u32) {
    if a + b < a {}
    //~^ panicking_overflow_checks
    if a > a + b {}
    //~^ panicking_overflow_checks
    if a + b < b {}
    //~^ panicking_overflow_checks
    if b > a + b {}
    //~^ panicking_overflow_checks
    if a - b > b {}
    if b < a - b {}
    if a - b > a {}
    //~^ panicking_overflow_checks
    if a < a - b {}
    //~^ panicking_overflow_checks
    if a + b < c {}
    if c > a + b {}
    if a - b < c {}
    if c > a - b {}
    let i = 1.1;
    let j = 2.2;
    if i + j < i {}
    if i - j < i {}
    if i > i + j {}
    if i - j < i {}
}

fn main() {
    test(1, 2, 3)
}
