#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct CustomMarker<'a>(PhantomData<&'a ()>);

impl<'a> Reborrow for CustomMarker<'a> {}

#[derive(Clone, Copy)]
struct StaticMarkerRef<'a>(PhantomData<&'a ()>);

impl<'a> CoerceShared<StaticMarkerRef<'static>> for CustomMarker<'a> {}
//~^ ERROR

fn method(_a: StaticMarkerRef<'static>) {}

fn main() {
    let a = CustomMarker(PhantomData);
    method(a);
    //~^ ERROR
}
