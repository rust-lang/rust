//@ run-pass
#![feature(stmt_expr_attributes)]

pub fn main() {
    let _x = #[inline(always)] || {};
    let _y = #[inline(never)] || {};
    let _z = #[inline] || {};
}
