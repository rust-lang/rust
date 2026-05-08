#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

#[derive(Reborrow)]
//~^ ERROR implementing `Reborrow` requires that a single lifetime parameter is passed between source and target
struct TooManyLifetimes<'a, 'b>(PhantomData<(&'a (), &'b ())>);

#[derive(Clone, Copy)]
struct BadTarget<'a>(&'a ());

#[derive(Reborrow, CoerceShared)]
//~^ ERROR the trait bound `&'a mut u32: CoerceShared<&'a ()>` is not satisfied
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
