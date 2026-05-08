#![feature(reborrow)]
use std::marker::{Reborrow, PhantomData};

#[derive(Reborrow)]
struct CustomMarker<'a>(PhantomData<&'a ()>);

fn method<'a>(_a: CustomMarker<'a>) -> &'a () {
    &()
}

fn main() {
    let a = CustomMarker(PhantomData);
    let b = method(a);
    let c = method(a);
    //~^ ERROR: cannot borrow `a` as mutable more than once at a time
    let _ = (b, c);
}
