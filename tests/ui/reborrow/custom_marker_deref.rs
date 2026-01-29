#![feature(reborrow)]
use std::ops::{Reborrow};
use std::marker::PhantomData;

#[derive(Debug)]
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
