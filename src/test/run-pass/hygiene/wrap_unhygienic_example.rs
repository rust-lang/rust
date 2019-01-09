// run-pass
// ignore-pretty pretty-printing is unhygienic

// aux-build:my_crate.rs
// aux-build:unhygienic_example.rs

#![feature(decl_macro)]

extern crate unhygienic_example;
extern crate my_crate; // (b)

// Hygienic version of `unhygienic_macro`.
pub macro hygienic_macro() {
    fn g() {} // (c)
    ::unhygienic_example::unhygienic_macro!();
    // ^ Even though we invoke an unhygienic macro, `hygienic_macro` remains hygienic.
    // In the above expansion:
    // (1) `my_crate` always resolves to (b) regardless of invocation site.
    // (2) The defined function `f` is only usable inside this macro definition.
    // (3) `g` always resolves to (c) regardless of invocation site.
    // (4) `$crate::g` remains hygienic and continues to resolve to (a).

    f();
}

#[allow(unused)]
fn test_hygienic_macro() {
    hygienic_macro!();

    fn f() {} // (d) no conflict
    f(); // resolves to (d)
}

fn main() {}
