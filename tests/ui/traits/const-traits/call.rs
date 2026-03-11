//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_closures, const_trait_impl)]
#![allow(incomplete_features)]

const _: () = {
    assert!((const || true)());
};

fn main() {}
