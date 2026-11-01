//@ run-pass

//! Test that CoerceShared with manually set lifetime bounds does allow moving a reborrowable type
//! after CoerceShared.

#![feature(reborrow)]
use std::marker::{CoerceShared, PhantomData, Reborrow};

#[derive(Reborrow)]
struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a: 'b, 'b> CoerceShared<CustomMarkerRef<'b>> for CustomMarker<'a> {}

#[derive(Clone, Copy)]
struct CustomMarkerRef<'a>(PhantomData<&'a ()>);

fn method<'a>(_a: CustomMarkerRef<'a>) -> &'a () {
    &()
}

fn move_into<T>(_: T) {}

fn main() {
    let a = CustomMarker(PhantomData);
    let _b = method(a);
    let _c = method(a);
    move_into(a);
}
