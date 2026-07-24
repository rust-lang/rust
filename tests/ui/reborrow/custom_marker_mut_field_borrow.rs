#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for CustomMarker<'a> {}

fn method<'a>(_a: CustomMarker<'a>) -> &'a () {
    &()
}

fn main() {
    let a = CustomMarker(PhantomData);
    let x = &a.0;
    let y = method(a);
    //~^ ERROR: cannot borrow `a` as mutable because it is also borrowed as immutable
    let _ = (x, y);
}
