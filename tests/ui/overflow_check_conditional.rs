#![warn(clippy::overflow_check_conditional)]
#![allow(clippy::needless_if)]

fn test(a: u32, b: u32, c: u32) {
    if a + b < a {} //~ overflow_check_conditional
    if a > a + b {} //~ overflow_check_conditional
    if a + b < b {} //~ overflow_check_conditional
    if b > a + b {} //~ overflow_check_conditional
    if a - b > b {}
    if b < a - b {}
    if a - b > a {} //~ overflow_check_conditional
    if a < a - b {} //~ overflow_check_conditional
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
