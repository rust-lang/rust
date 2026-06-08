#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

fn method<'a>(_a: CustomMarker<'a>) -> &'a () {
    &()
}

fn main() {
    let a = CustomMarker(PhantomData);
    let b = method(a);
    let _ = method(a);
    //~^ ERROR: cannot borrow `a` as mutable more than once at a time
    let _ = (a, b);
    //~^ ERROR: cannot move out of `a` because it is borrowed
}
