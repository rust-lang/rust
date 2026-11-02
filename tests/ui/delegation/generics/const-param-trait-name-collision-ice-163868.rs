#![feature(fn_delegation)]

trait C {}

struct S<const C: usize>;

impl<const C: usize> S<C> {
    //~^ ERROR type provided when a constant was expected
    fn foo_self<T>(self) {}
}

trait Trait {
    reuse S::<1>::foo_self::<_> as foo_self_4;
    //~^ ERROR failed to inherit delegation predicates
    //~| ERROR the size for values of type `T` cannot be known
}

fn main() {}
