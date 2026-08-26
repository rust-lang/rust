//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args, min_generic_const_args)]

struct S;
const C: S = S;

trait Trait {
    const F: S;
}

fn take(_: impl Trait<F = { core::gca!(C) }>) {}
//~^ ERROR `S` must implement `ConstParamTy`

fn main() {}
