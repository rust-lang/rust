#![feature(reborrow)]
use std::ops::{CoerceShared, Reborrow};
use std::marker::PhantomData;

#[derive(Debug)]
struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}
#[derive(Debug, Clone, Copy)]
struct CustomMarkerRef<'a>(PhantomData<&'a ()>);
impl<'a> CoerceShared<CustomMarkerRef<'a>> for CustomMarker<'a> {}


fn method<'a>(_a: CustomMarkerRef<'a>) -> &'a () {
    &()
}

fn main() {
    let a = CustomMarker(PhantomData);
    let b = method(a);
    let c = method(a);
    let _ = (&a, b, c);
}
