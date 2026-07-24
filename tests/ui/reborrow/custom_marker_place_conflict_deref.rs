//@ check-fail

#![feature(reborrow)]
use std::{marker::{Reborrow, PhantomData}, ops::Deref};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

impl<'a> Deref for CustomMarker<'a> {
    type Target = ();

    fn deref(&self) -> &() {
        &()
    }
}

fn reborrow(_: CustomMarker) {}

fn main() {
    let a = CustomMarker(PhantomData);
    let b: &() = &a;
    reborrow(a);
    //~^ ERROR cannot borrow `a` as mutable because it is also borrowed as immutable
    let _ = b;
}
