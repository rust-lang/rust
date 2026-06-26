//@ known-bug: unknown
//@ edition: 2024

#![feature(reborrow)]

// Malformed no-ICE regression: the unrelated name and signature errors are intentional because the
// original issue combined them with CoerceShared validation.

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct CustomMarker<'a>(PhantomData<&'a ()>);

struct CustomMarkerRef<'a>(PhantomData<(Debug, Clone, Copy)>);
//~^ ERROR
//~| ERROR
//~| ERROR

impl<'a> Reborrow for CustomMarker<'a> {}
impl<'a> CoerceShared<CustomMarkerRef<'a>> for CustomMarker<'a> {}
//~^ ERROR

fn method<'a>(_a: CustomMarkerRef<'a>) -> 'a () {
    //~^ ERROR
    &()
}

fn main() {
    let a = CustomMarker(PhantomData);
    let b = method(a);
}
