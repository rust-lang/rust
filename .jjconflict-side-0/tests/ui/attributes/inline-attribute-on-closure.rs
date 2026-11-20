//! Regression test for https://github.com/rust-lang/rust/issues/49632

//@ run-pass
#![feature(stmt_expr_attributes)]

pub fn main() {
    let _x = #[inline(always)] || {};
    let _y = #[inline(never)] || {};
    let _z = #[inline] || {};
}
