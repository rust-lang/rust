#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

#[derive(Reborrow, CoerceShared)]
//~^ ERROR `derive(CoerceShared)` requires exactly one `#[coerce_shared(Target)]` attribute
struct MissingTarget<'a>(&'a mut ());

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(Target<'a>)]
#[coerce_shared(Target<'a>)]
//~^ ERROR duplicate `#[coerce_shared(Target)]` attribute for `derive(CoerceShared)`
struct DuplicateTarget<'a>(&'a mut ());

#[derive(Reborrow, CoerceShared)]
#[coerce_shared]
//~^ ERROR malformed `#[coerce_shared(Target)]` attribute for `derive(CoerceShared)`
struct MalformedTargetWord<'a>(&'a mut ());

#[derive(Reborrow, CoerceShared)]
#[coerce_shared()]
//~^ ERROR malformed `#[coerce_shared(Target)]` attribute for `derive(CoerceShared)`
struct MalformedTargetEmpty<'a>(&'a mut ());

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(Target<'a>, Target<'a>)]
//~^ ERROR malformed `#[coerce_shared(Target)]` attribute for `derive(CoerceShared)`
struct MalformedTargetList<'a>(&'a mut ());

#[derive(Clone, Copy)]
struct Target<'a>(&'a ());

fn main() {}
