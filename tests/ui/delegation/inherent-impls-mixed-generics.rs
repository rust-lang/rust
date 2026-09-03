#![feature(fn_delegation)]

struct S<'a, A, B, const C: usize> {
    xd: &'a [(A, B); C],
}

impl<'a, 'b, 'c, A, const C: usize> S<'static, A, usize, C> {
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

trait Trait<'a, AA, BB> where Self: Sized {
    reuse S::foo_self;
    //~^ ERROR: inferred lifetimes are not allowed in delegations as we need to inherit signature
    //~| ERROR: the placeholder `_` is not allowed within types on item signatures for associated functions
    //~| ERROR: the placeholder `_` is not allowed within types on item signatures for associated functions
    //~| ERROR: the placeholder `_` is not allowed within types on item signatures for associated functions
    //~| ERROR: this function takes 1 argument but 0 arguments were supplied
}

fn main() {}
