//@ edition: 2024

// A `CoerceShared` target without a reborrowed lifetime must be rejected without an ICE.

#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct CustomMarker<'a>(PhantomData<&'a ()>);
struct CustomMarkerRef;

impl<'a> Reborrow for CustomMarker<'a> {}
impl<'a> CoerceShared<CustomMarkerRef> for CustomMarker<'a> {}
//~^ ERROR implementing `CoerceShared` does not allow multiple lifetimes or fields to be coerced

fn method(_a: CustomMarkerRef) {}

fn main() {
    let a = CustomMarker(PhantomData);
    method(a);
}
