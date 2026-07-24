//! Test that CoerceShared of custom ZST marker type reborrows the type automatically as shared but
//! moving the original invalidates the results.

#![feature(reborrow)]
use std::marker::{CoerceShared, PhantomData, Reborrow};

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(CustomMarkerRef<'a>)]
struct CustomMarker<'a>(PhantomData<&'a ()>);
#[derive(Clone, Copy)]
struct CustomMarkerRef<'a>(PhantomData<&'a ()>);

fn method<'a>(_a: CustomMarkerRef<'a>) -> &'a () {
    &()
}

fn main() {
    let a = CustomMarker(PhantomData);
    let b = method(a);
    let c = method(a);
    let _ = (a, b, c);
    //~^ ERROR: cannot move out of `a` because it is borrowed
}
