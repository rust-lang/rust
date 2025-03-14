//@ check-pass
//@edition:2018

#![allow(redundant_semicolons, clippy::no_effect)]
#![feature(stmt_expr_attributes)]

#[rustfmt::skip]
fn main() {
    #[clippy::author]
    {
        let x = 42i32;
        let _t = 1f32;

        -x;
    };
    #[clippy::author]
    {
        let expr = String::new();
        drop(expr)
    };

    #[clippy::author]
    async move || {};
}
