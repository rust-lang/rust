//@ edition: 2024

#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct CustomMarker<'a>(PhantomData<&'a ()>);
struct CustomMarkerRef;

impl<'a> Reborrow for CustomMarker<'a> {}
impl<'a> CoerceShared<CustomMarkerRef> for CustomMarker<'a> {}
//~^ ERROR

fn method(_a: CustomMarkerRef) {}

fn main() {
    let a = CustomMarker(PhantomData);
    method(a);
    //~^ ERROR
}
