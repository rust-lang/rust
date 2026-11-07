//! Test that reborrowing a custom marker type conflicts with a reference to a field of the type.

#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

fn reborrow(_: CustomMarker) {}

fn main() {
    let a = CustomMarker(PhantomData);
    let b: &PhantomData<&()> = &a.0;
    reborrow(a);
    //~^ ERROR cannot borrow `a` as mutable because it is also borrowed as immutable
    let _ = b;
}
