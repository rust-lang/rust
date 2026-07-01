//@ run-pass

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
