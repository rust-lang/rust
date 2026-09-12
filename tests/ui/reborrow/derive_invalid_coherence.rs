//! Test that deriving Reborrow errors on multiple lifetimes and non-Copy/Reborrow fields, and that
//! CoerceShared errors if field types do not match.

#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

#[derive(Reborrow)]
//~^ ERROR implementing `Reborrow` requires that a single lifetime parameter is passed between source and target
struct TooManyLifetimes<'a, 'b>(PhantomData<(&'a (), &'b ())>);

#[derive(Clone, Copy)]
struct BadTarget<'a>(&'a ());
//~^ ERROR implementing `CoerceShared` requires corresponding fields to match

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(BadTarget<'a>)]
struct BadSource<'a>(&'a mut u32);

struct NotCopy;

#[derive(Reborrow)]
struct NotCopyField<'a> {
    field: NotCopy,
    //~^ ERROR the trait bound `NotCopy: Copy` is not satisfied
    marker: PhantomData<&'a ()>,
}

fn main() {}
