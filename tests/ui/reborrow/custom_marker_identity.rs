//@ check-fail

#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

fn method<'a>(a: CustomMarker<'a>) -> CustomMarker<'a> { //~ERROR cannot return reference to temporary value
    //~^ ERROR cannot return value referencing function parameter `a`
    a
}

fn main() {
    let a = CustomMarker(PhantomData);
    let _ = method(a);
}
