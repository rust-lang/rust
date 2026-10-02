//! Test that reborrowing a custom marker type conflicts with an earlier reborrow even if the result
//! is immediately dropped.

#![feature(reborrow)]
use std::marker::{PhantomData, Reborrow};

#[derive(Reborrow)]
struct CustomMarker<'a>(PhantomData<&'a ()>);

fn method<'a>(_a: CustomMarker<'a>) -> &'a () {
    &()
}

fn main() {
    let a = CustomMarker(PhantomData);
    let b = method(a);
    let _ = method(a);
    //~^ ERROR: cannot borrow `a` as mutable more than once at a time
    let _ = (a, b);
    //~^ ERROR: cannot borrow `a` as mutable more than once at a time [E0499]
}
