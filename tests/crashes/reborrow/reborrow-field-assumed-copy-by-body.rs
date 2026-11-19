//@ known-bug: unknown
//@ edition: 2024

#![feature(reborrow)]

// `Inner` has no `Copy` impl, so the `Reborrow` impl check counts `inner` as the
// reborrowed field of `W`. `f` assumes `Inner<'a, T>: Copy` with a where clause, so
// borrowck copies `inner` there, finds no reborrowed field and hits the
// single-lifetime `bug!` in `gather_reborrows`. Move this to `tests/ui/reborrow`
// once it is fixed.

use std::marker::{PhantomData, Reborrow};

struct Inner<'a, T>(&'a mut (), PhantomData<T>);

impl<'a, T> Reborrow for Inner<'a, T> {}

struct W<'a, 'b, T> {
    inner: Inner<'a, T>,
    shared: &'b (),
}

impl<'a, 'b, T> Reborrow for W<'a, 'b, T> {}

pub fn f<'a, 'b, T>(x: W<'a, 'b, T>) -> W<'a, 'b, T>
where
    Inner<'a, T>: Copy,
{
    x
}

fn main() {}
