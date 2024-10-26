// FIXME(effects) check-pass
//@ compile-flags: -Znext-solver
#![feature(const_closures, const_trait_impl, effects)]
#![allow(incomplete_features)]

pub const _: () = {
    assert!((const || true)());
    //~^ ERROR cannot call non-const closure in constants
};

fn main() {}
