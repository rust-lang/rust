//@ known-bug: unknown
//@ edition: 2024

#![feature(reborrow)]

// Coherence checks whether `w` is `Copy` with the impl's bounds, where it is not, so it
// counts as reborrowed and the impl is accepted. Borrowck checks with the bounds of `f`,
// where `T: Copy` makes `w` `Copy`, finds no reborrowed field, and hits the
// single-lifetime `bug!` in `gather_reborrows`. Move this to `tests/ui/reborrow` once
// it is fixed. Follow-up to #165653.

use std::marker::{PhantomData, Reborrow};

#[derive(Clone, Copy)]
struct Wrap<'a, T>(PhantomData<&'a ()>, T);

impl<'a, T: Reborrow> Reborrow for Wrap<'a, T> {}

struct W<'a, 'b, T> {
    w: Wrap<'a, T>,
    s: &'b (),
}

impl<'a, 'b, T: Reborrow> Reborrow for W<'a, 'b, T> {}

pub fn f<'a, 'b, T: Reborrow + Copy>(x: W<'a, 'b, T>) -> W<'a, 'b, T> {
    x
}

fn main() {}
