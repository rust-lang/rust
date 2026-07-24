//@ check-fail

//! Check that the result of a Reborrow retains the original lifetime and does not capture local
//! values, therefore enabling an identity function to compile.
//! This should eventually pass.

#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

fn method<'a>(a: CustomMarker<'a>) -> CustomMarker<'a> { //~ERROR cannot return reference to temporary value
    //~^ ERROR cannot return value referencing local data `a`
    a
}

fn main() {
    let a = CustomMarker(PhantomData);
    let _ = method(a);
}
