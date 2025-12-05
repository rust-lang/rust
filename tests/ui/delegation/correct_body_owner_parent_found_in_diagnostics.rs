#![feature(fn_delegation)]
#![allow(incomplete_features)]

use std::marker::PhantomData;

pub struct InvariantRef<'a, T: ?Sized>(&'a T, PhantomData<&'a mut &'a T>);

impl<'a> InvariantRef<'a, ()> {
    pub const NEW: Self = InvariantRef::new(&());
    //~^ ERROR: no function or associated item named `new` found
}

trait Trait {
    fn foo(&self) -> u8 { 0 }
    fn bar(&self) -> u8 { 1 }
    fn meh(&self) -> u8 { 2 }
}

struct Z(u8);

impl Trait for Z {
    reuse <u8 as Trait>::{foo, bar, meh} { &const { InvariantRef::<'a>::NEW } }
    //~^ ERROR: use of undeclared lifetime name `'a`
    //~| ERROR: use of undeclared lifetime name `'a`
    //~| ERROR: use of undeclared lifetime name `'a`
    //~| ERROR: the trait bound `u8: Trait` is not satisfied
    //~| ERROR: the trait bound `u8: Trait` is not satisfied
    //~| ERROR: the trait bound `u8: Trait` is not satisfied
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
    //~| ERROR: mismatched types
}

fn main() { }
