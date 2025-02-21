#![warn(clippy::as_underscore)]

fn foo(_n: usize) {}

fn main() {
    let n: u16 = 256;
    foo(n as _);
    //~^ as_underscore

    let n = 0_u128;
    let _n: u8 = n as _;
    //~^ as_underscore
}
