//@ check-fail

#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMut<'a>(&'a mut ());
impl<'a> Reborrow for CustomMut<'a> {}

fn reborrow(_: CustomMut) {}

fn main() {
    let a = CustomMut(&mut ());
    let b: &CustomMut = &a;
    reborrow(a);
    //~^ ERROR cannot borrow `a` as mutable because it is also borrowed as immutable
    let _ = b;
}
