//@ run-pass

//! Test that CoerceShared of custom ZST marker type reborrows the type automatically as shared and
//! the original stays concurrently usable through shared references.

#![feature(reborrow)]
use std::marker::{CoerceShared, PhantomData, Reborrow};

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(CustomMarkerRef<'a>)]
struct CustomMarker<'a>(PhantomData<&'a ()>);
#[derive(Debug, Clone, Copy)]
struct CustomMarkerRef<'a>(PhantomData<&'a ()>);

fn method<'a>(_a: CustomMarkerRef<'a>) -> &'a () {
    &()
}

fn main() {
    let a = CustomMarker(PhantomData);
    let b = method(a);
    let c = method(a);
    let _ = (&a, b, c);
}
