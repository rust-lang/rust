// FIXME(const_trait_impl) check-pass
//@ compile-flags: -Znext-solver
#![feature(const_closures, const_trait_impl)]
#![allow(incomplete_features)]

pub const _: () = {
    assert!((const || true)());
    //~^ ERROR }: [const] Fn()` is not satisfied
};

fn main() {}
