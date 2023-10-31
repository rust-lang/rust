//! This test checks the behaviour of walking into binders
//! and normalizing something behind them actually works.

// edition: 2021

#![feature(type_alias_impl_trait)]

trait B {
    type C;
}

struct A;

impl<'a> B for &'a A {
    type C = Tait;
}

type Tait = impl std::fmt::Debug;

struct Terminator;

type Successors<'a> = impl std::fmt::Debug + 'a;

impl Terminator {
    fn successors(&self, mut f: for<'x> fn(&'x ()) -> <&'x A as B>::C) -> Successors<'_> {
        f = g;
        //~^ ERROR item constrains opaque type that is not in its signature
    }
}

fn g(_: &()) -> String {
    String::new()
}

fn main() {}
