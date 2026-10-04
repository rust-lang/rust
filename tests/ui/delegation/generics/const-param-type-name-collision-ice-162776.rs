#![feature(fn_delegation)]

struct S<A, const C: usize>;
//~^ ERROR: type parameter `A` is never used

impl<'b, 'c, C, const C: usize> S<C, C> {
//~^ ERROR: the name `C` is already used for a generic parameter in this item's generic parameters
//~| ERROR: type provided when a constant was expected
    fn foo_self<T, const B: bool>() {}
}

reuse S::<(), 123>::foo_self;

fn main() {}
