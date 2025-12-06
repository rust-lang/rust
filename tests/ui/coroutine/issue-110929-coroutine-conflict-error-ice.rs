//@ edition:2021
//@ check-pass
#![feature(coroutines, stmt_expr_attributes)]

fn main() {
    let x = &mut ();
    || {
        let _c = #[coroutine]
        || (*&mut *x).yield;
        || _ = &mut *x;
    };
}
