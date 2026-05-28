//@ run-pass

#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

fn method<'a>(_a: CustomMarker<'a>) -> &'a () {
    &()
}

fn main() {
    let a = CustomMarker(PhantomData);
    let _ = method(a);
    let _ = method(a);
}
