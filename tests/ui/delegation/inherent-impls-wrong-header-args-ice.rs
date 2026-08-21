#![feature(fn_delegation)]

struct S<'a, A, const C: usize> {
    xd: &'a [A; C],
}

impl<'a, 'b, 'c, A, const C: usize> S<A, C> {
//~^ ERROR: implicit elided lifetime not allowed here
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

trait Trait<'a, AA, BB> where Self: Sized {
    reuse S::<(), ()>::foo_self;
    //~^ ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
    //~| ERROR: type provided when a constant was expected
    //~| ERROR: type provided when a constant was expected
}

fn main() {}
