//! Regression test for <https://github.com/rust-lang/rust/issues/153831>
//@ check-fail
//@compile-flags: -Znext-solver=globally --emit=obj
#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

type const A: () = A;
//~^ ERROR: overflow evaluating the requirement `A == _`
//~| ERROR: overflow evaluating the requirement `A == _`
//~| ERROR: type annotations needed

fn main() {
    A;
}
