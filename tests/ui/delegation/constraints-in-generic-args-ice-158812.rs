#![feature(fn_delegation)]

pub trait Trait<'a> {
    fn foo(&self);
}

pub struct S<'a, A>(&'a A);
impl<'a, A> Trait<'a> for S<'a, A> {
    reuse Trait::<A = ()>::foo;
    //~^ ERROR: associated item constraints are not allowed here
}

fn main() {}
