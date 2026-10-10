//@ run-pass

//! Test that CoerceShared of custom ZST marker type reborrows the type automatically as shared but
//! moving the original is possible afterwards if the shared results do not remain alive.

#![feature(reborrow)]
use std::marker::{CoerceShared, PhantomData, Reborrow};

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(CustomMarkerRef<'a>)]
struct CustomMarker<'a>(PhantomData<&'a ()>);
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
