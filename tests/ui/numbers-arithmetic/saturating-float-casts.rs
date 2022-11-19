// run-pass
// compile-flags:-Zmir-opt-level=0 -Zmir-enable-passes=-CheckNiches

#![feature(test, stmt_expr_attributes)]
#![deny(overflowing_literals)]

#[path = "saturating-float-casts-impl.rs"]
mod implementation;

pub fn main() {
    implementation::run();
}
