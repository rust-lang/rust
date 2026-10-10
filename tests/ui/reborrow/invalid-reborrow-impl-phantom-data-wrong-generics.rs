//! Regression test for an ICE in borrowck when a type is reborrowed through a `Reborrow` impl
//! that coherence has already rejected.
//! See <https://github.com/rust-lang/rust/issues/156308>.

#![feature(reborrow)]
use std::marker::{PhantomData, Reborrow};

struct CustomMarker<'a>(PhantomData<&'a ()>);
impl<'a> Reborrow for PhantomData<'a> {}
//~^ ERROR E0107
//~| ERROR E0107
//~| ERROR implementing `Reborrow` requires exactly one lifetime argument in the reborrowed type

fn main() {
    let mut a = CustomMarker(PhantomData);
}
