//@ compile-flags: -Znext-solver=globally

#![feature(gca_const_items, gca_min_const_items)]

struct S;
const C: S = S;

trait Trait {
    const F: S;
}

fn take(_: impl Trait<F = { core::gca!(C) }>) {}
//~^ ERROR `S` must implement `ConstParamTy`

fn main() {}
