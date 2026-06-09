//! This test checks that we can't actually have an opaque type behind
//! a binder that references variables from that binder.

//@ edition: 2021

#![feature(type_alias_impl_trait)]

trait B {
    type C;
}

struct A;

impl<'a> B for &'a A {
    type C = Tait<'a>;
}

type Tait<'a> = impl std::fmt::Debug + 'a;

struct Terminator;

type Successors<'a> = impl std::fmt::Debug + 'a;

impl Terminator {
    #[define_opaque(Successors, Tait)]
    fn successors(&self, mut f: for<'x> fn(&'x ()) -> <&'x A as B>::C) -> Successors<'_> {
        f = g; //~ ERROR expected generic lifetime parameter, found `'x`
    }
}

fn g(x: &()) -> &() {
    x
}

fn main() {}
