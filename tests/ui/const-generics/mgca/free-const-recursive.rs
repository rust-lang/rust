//! Regression test for <https://github.com/rust-lang/rust/issues/153831>
//@ check-fail
//@compile-flags: -Znext-solver=globally --emit=obj
#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

type const A: () = A;
//~^ ERROR type mismatch resolving `A normalizes-to _`
//~| ERROR the constant `A` is not of type `()`

fn main() {
    A;
}
