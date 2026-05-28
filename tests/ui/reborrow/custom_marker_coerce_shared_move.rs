#![feature(reborrow)]
use std::marker::{CoerceShared, PhantomData, Reborrow};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}
#[derive(Clone, Copy)]
struct CustomMarkerRef<'a>(PhantomData<&'a ()>);
impl<'a> CoerceShared<CustomMarkerRef<'a>> for CustomMarker<'a> {}


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
