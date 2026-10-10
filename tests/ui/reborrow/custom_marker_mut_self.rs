//! Test that reborrowing a custom marker type implementing Reborrow conflicts with moving the type.

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
    let _ = (a, b); //~ERROR cannot borrow `a` as mutable more than once at a time [E0499]
}
