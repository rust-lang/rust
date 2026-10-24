//! Test that CoerceShared with manually set lifetime bounds does allow moving a reborrowable type
//! after CoerceShared.
//! This should probably work eventually, but right now it fails from trait well-formedness checks.

#![feature(reborrow)]
use std::marker::{CoerceShared, PhantomData, Reborrow};

#[derive(Reborrow)]
struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a: 'b, 'b> CoerceShared<CustomMarkerRef<'b>> for CustomMarker<'a> {}
//~^ ERROR: implementing `CoerceShared` requires that a single lifetime parameter is passed between source and target

#[derive(Clone, Copy)]
struct CustomMarkerRef<'a>(PhantomData<&'a ()>);

fn method<'a>(_a: CustomMarkerRef<'a>) -> &'a () {
    &()
}

fn move_into<T>(_: T) {}

fn main() {
    let a = CustomMarker(PhantomData);
    let b = method(a);
    let c = method(a);
    move_into(a);
}
