//! This test checks that walking into binders
//! during opaque type collection does not ICE or raise errors.

// edition: 2021

// check-pass

#![feature(type_alias_impl_trait)]

trait B {
    type C;
}

struct A;

impl<'a> B for &'a A {
    type C = ();
}

struct Terminator;

type Successors<'a> = impl std::fmt::Debug + 'a;

impl Terminator {
    fn successors(&self, _: for<'x> fn(&'x ()) -> <&'x A as B>::C) -> Successors<'_> {}
}

fn main() {}
