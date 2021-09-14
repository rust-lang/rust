#![warn(clippy::overflow_check_conditional)]

fn main() {
    let a: u32 = 1;
    let b: u32 = 2;
    let c: u32 = 3;
    if a + b < a {}
    if a > a + b {}
    if a + b < b {}
    if b > a + b {}
    if a - b > b {}
    if b < a - b {}
    if a - b > a {}
    if a < a - b {}
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
