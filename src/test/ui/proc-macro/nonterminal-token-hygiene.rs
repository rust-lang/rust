// Make sure that marks from declarative macros are applied to tokens in nonterminal.

// check-pass
// aux-build:test-macros.rs

#![feature(decl_macro)]

#[macro_use]
extern crate test_macros;

macro_rules! outer {
    ($item:item) => {
        macro inner() {
            recollect! { $item }
        }

        inner!();
    };
}

struct S;

outer! {
    struct S; // OK, not a duplicate definition of `S`
}

fn main() {}
