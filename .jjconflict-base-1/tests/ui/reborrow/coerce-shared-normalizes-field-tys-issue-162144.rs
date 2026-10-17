//@ check-pass

#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(CustomMarkerRef<'a>)]
struct CustomMarker<'a>(PhantomData<&'a ()>);

#[derive(Clone, Copy)]
struct CustomMarkerRef<'a>(PhantomData<&'a <Self as Iterator>::Item>);

impl<'a> Iterator for CustomMarkerRef<'a> {
    type Item = ();

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn method<'a>(_a: CustomMarkerRef<'a>) -> &'a () {
    &()
}

fn main() {
    let a = CustomMarker(PhantomData);
    let _b = method(a);
}
