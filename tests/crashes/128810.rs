//@ known-bug: rust-lang/rust#128810

#![feature(fn_delegation)]

use std::marker::PhantomData;

pub struct InvariantRef<'a, T: ?Sized>(&'a T, PhantomData<&'a mut &'a T>);

impl<'a> InvariantRef<'a, ()> {
    pub const NEW: Self = InvariantRef::new(&());
}

trait Trait {
    fn foo(&self) -> u8 { 0 }
    fn bar(&self) -> u8 { 1 }
    fn meh(&self) -> u8 { 2 }
}

struct Z(u8);

impl Trait for Z {
    reuse <u8 as Trait>::{foo, bar, meh} { &const { InvariantRef::<'a>::NEW } }
}

fn main() { }
