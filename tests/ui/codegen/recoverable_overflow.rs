//@ no-prefer-dynamic
//@ compile-flags: -Coverflow-checks=recoverable
//@ run-pass
//@ check-run-results
#![feature(recoverable_integer_overflow)]
#![allow(arithmetic_overflow, unused)]

#[core::panic::integer_overflow_action]
fn overflow() {
    println!("overflow happened")
}

fn main() {
    let mut x = 255u8;
    x += 1;
}
