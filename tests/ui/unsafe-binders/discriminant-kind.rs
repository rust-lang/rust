// Regression test for https://github.com/rust-lang/rust/issues/158839
//@ check-pass
//@ revisions: current next
//@ [next] compile-flags: -Znext-solver=globally

#![feature(unsafe_binders)]

fn foo<T: std::marker::Copy>(x: unsafe<> T) {
    std::mem::discriminant(&x);
}

fn main() {}
