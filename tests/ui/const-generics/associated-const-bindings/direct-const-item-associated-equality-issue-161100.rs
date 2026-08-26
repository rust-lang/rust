//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args, min_generic_const_args)]
#![allow(incomplete_features)]

struct S;
const C: S = S;

trait Trait {
    const F: S;
}

fn take(_: impl Trait<F = { core::direct_const_arg!(C) }>) {}
//~^ ERROR `S` must implement `ConstParamTy`

fn main() {}
