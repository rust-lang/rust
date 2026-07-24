//@ run-pass

//! Test that CoerceShared of custom ZST marker type reborrows the type automatically from a
//! `&mut CustomMarker` deref.

#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

fn method<'a>(_a: CustomMarker<'a>) -> &'a () {
    &()
}

fn main() {
    let mut a = CustomMarker(PhantomData);
    let b = &mut a;
    let _ = method(*b);
}
