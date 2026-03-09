//! Make sure that the closure still gets the correct precedence when round-tripping
//! through a proc macro.
//! The correct precendence is `(|| ()) as fn()`, even though these parentheses are not
//! directly part of the code.
//! If it would get lost, the code would be `|| () as fn()`, get parsed as
//! `|| (() as fn())` and fail to compile.
//! Notably, this will also fail to compile if we use `recollect` instead of `identity`.
//! Regression test for https://github.com/rust-lang/rust/pull/151830#issuecomment-4010899019.
//@ proc-macro: test-macros.rs
//@ check-pass

macro_rules! operator_impl {
    ($target_expr:expr) => {
        test_macros::identity! {
            $target_expr as fn()
        };
    };
}

fn main() {
    operator_impl!(|| ());
}
