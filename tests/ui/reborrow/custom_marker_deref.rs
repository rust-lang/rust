//@ run-pass

#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

#[derive(Reborrow)]
struct CustomMarker<'a>(PhantomData<&'a ()>);

fn method<'a>(_a: CustomMarker<'a>) -> &'a () {
    &()
}

fn main() {
    let mut a = CustomMarker(PhantomData);
    let b = &mut a;
    let _ = method(*b);
}
