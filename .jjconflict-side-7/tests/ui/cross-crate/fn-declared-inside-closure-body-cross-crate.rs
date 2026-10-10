//@ run-pass
//@ aux-build:fn-declared-inside-closure-body-cross-crate.rs

//! Regression test for https://github.com/rust-lang/rust/issues/2723

extern crate fn_declared_inside_closure_body_cross_crate;
use fn_declared_inside_closure_body_cross_crate::f;

pub fn main() {
    unsafe {
        f(vec![2]);
    }
}
