//! Test that CoerceShared cannot be implemented with spurious 'static lifetimes.

#![feature(reborrow)]

// The impl is accepted, but using it to coerce a local marker into a `'static`
// target still requires the local borrow to live for `'static`.

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct CustomMarker<'a>(PhantomData<&'a ()>);

impl<'a> Reborrow for CustomMarker<'a> {}

#[derive(Clone, Copy)]
struct StaticMarkerRef<'a>(PhantomData<&'a ()>);

// Should error: for two types with only one lifetime each, both should use the same lifetime.
impl<'a> CoerceShared<StaticMarkerRef<'static>> for CustomMarker<'a> {}

fn method(_a: StaticMarkerRef<'static>) {}

fn main() {
    let a = CustomMarker(PhantomData);
    method(a);
    //~^ ERROR
}
