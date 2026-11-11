//! Regression test for <https://github.com/rust-lang/rust/issues/21922>.
//!
//! Ensure Add works with all value/reference operand combinations,
//! both via the + operator and ufcs.
//!
//! Originally method lookup failed only for x + &y.

//@ run-pass
use std::ops::Add;
fn show(z: i32) {
    println!("{}", z)
}
fn main() {
    let x = 23;
    let y = 42;
    show(Add::add( x,  y));
    show(Add::add( x, &y));
    show(Add::add(&x,  y));
    show(Add::add(&x, &y));
    show( x +  y);
    show( x + &y);
    show(&x +  y);
    show(&x + &y);
}
