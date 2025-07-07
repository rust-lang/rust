#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

type NonNullU32_2 = pattern_type!(u32 is 1..=);
//~^ ERROR: inclusive range with no end
type Positive2 = pattern_type!(i32 is 0..=);
//~^ ERROR: inclusive range with no end
type Wild = pattern_type!(() is _);
//~^ ERROR: pattern not supported in pattern types

// FIXME: confusing diagnostic because `not` can be a binding
type NonNull = pattern_type!(*const () is not null);
//~^ ERROR: expected one of `@` or `|`, found `null`
//~| ERROR: pattern not supported in pattern types

type NonNull2 = pattern_type!(*const () is !nil);
//~^ ERROR: expected `null`, found `nil`

// FIXME: reject with a type mismatch
type Mismatch2 = pattern_type!(() is !null);

fn main() {}
