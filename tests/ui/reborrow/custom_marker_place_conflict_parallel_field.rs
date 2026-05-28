//@ check-fail

#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

struct CustomMarkerTwo<'a>(CustomMarker<'a>, u64);
impl<'a> Reborrow for CustomMarkerTwo<'a> {}

fn reborrow(_: CustomMarkerTwo) {}

fn main() {
    let a = CustomMarker(PhantomData);
    let a = CustomMarkerTwo(a, 0);
    let b: &u64 = &a.1;
    reborrow(a);
    //~^ ERROR cannot borrow `a` as mutable because it is also borrowed as immutable
    let _ = b;
}
