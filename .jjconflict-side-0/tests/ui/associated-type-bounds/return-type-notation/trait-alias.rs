// Regression test for <https://github.com/rust-lang/rust/issues/152158>.
//@ check-pass
#![feature(return_type_notation, trait_alias)]

trait Tr {
    fn f() -> impl Sized;
}

trait Al = Tr;

fn f<T: Al<f(..): Copy>>() {}

fn main() {}
