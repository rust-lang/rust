//@ check-pass
// Regression test for #5238 / https://github.com/rust-lang/rust/pull/69562

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

fn main() {
    let _ = #[coroutine]
    || {
        yield;
    };
}
