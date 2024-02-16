//@ run-pass
//@ compile-flags:-Zmir-opt-level=0

#![feature(test, stmt_expr_attributes)]
#![deny(overflowing_literals)]

#[path = "saturating-float-casts-impl.rs"]
mod implementation;

pub fn main() {
    implementation::run();
}
