//! Regression test for <https://github.com/rust-lang/rust/issues/165604>.
//!
//! Reborrowing a `Reborrow` type through a shared reference used to cause
//! an ICE in borrowck instead of an error.

#![feature(reborrow)]

use std::marker::{PhantomData, Reborrow};

#[derive(Reborrow)]
struct CustomMarker<'a>(PhantomData<&'a ()>);

fn reborrow(_: CustomMarker) {}

fn main() {
    let a = CustomMarker(PhantomData);
    let b = &a;
    reborrow(*b);
    //~^ ERROR cannot borrow `*b` as mutable, as it is behind a `&` reference
}
