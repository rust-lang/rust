//! Test that reborrowing a custom mut type conflicts with a reborrow of the &mut it wraps.

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
