//@ check-pass

//@ aux-build:my_crate.rs
//@ aux-build:unhygienic_example.rs

#![feature(decl_macro)]

extern crate unhygienic_example;
extern crate my_crate; // (b)

use unhygienic_example::g;

// Hygienic version of `unhygienic_macro`.
pub macro hygienic_macro() {
    ::unhygienic_example::unhygienic_macro!();

    f();
}

#[allow(unused)]
fn test_hygienic_macro() {
    hygienic_macro!();

    fn f() {} // (d) no conflict
    f(); // resolves to (d)
}

fn main() {}
