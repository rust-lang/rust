//@ run-pass

#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

#[derive(Clone, Copy)]
struct CustomMarkerRef<'a>(PhantomData<&'a ()>);
impl<'a> CoerceShared<CustomMarkerRef<'a>> for CustomMarker<'a> {}

fn take_mut<'a>(_value: CustomMarker<'a>) {}
fn take_ref<'a>(_value: CustomMarkerRef<'a>) {}

fn assignment_style_reborrow(value: CustomMarker<'_>) {
    let reborrowed = value;
    take_mut(reborrowed);
}

fn assignment_style_coerce_shared(value: CustomMarker<'_>) {
    let shared: CustomMarkerRef<'_> = value;
    take_ref(shared);
}

fn rvalue_style_coerce_shared(value: CustomMarker<'_>) {
    let mut slots = [CustomMarkerRef(PhantomData)];
    slots[0] = value;
    take_ref(slots[0]);
}

fn main() {
    assignment_style_reborrow(CustomMarker(PhantomData));
    assignment_style_coerce_shared(CustomMarker(PhantomData));
    rvalue_style_coerce_shared(CustomMarker(PhantomData));
}
