#![feature(fn_delegation)]

struct S<'a, A, const C: usize> {
    xd: &'a [A; C],
}

impl<'a, 'b, 'c, A, const C: usize> S<A, C> {
//~^ ERROR: implicit elided lifetime not allowed here
    fn foo_self<'d: 'd, 'e, T, const B: bool>(self) {}
}

trait Trait<'a, AA, BB> where Self: Sized {
    reuse S::foo_self;
    //~^ ERROR: this function takes 1 argument but 0 arguments were supplied
}

fn main() {}
