//@ check-fail

#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMut<'a>(&'a mut ());
impl<'a> Reborrow for CustomMut<'a> {}

fn reborrow(_: CustomMut) {}

fn main() {
    let a = CustomMut(&mut ());
    let b: &mut () = a.0;
    reborrow(a);
    //~^ ERROR cannot borrow `a` as mutable more than once at a time
    let _ = b;
}
