//@ compile-flags: -Zno-analysis

#![feature(pattern_type_macro)]

use std::pat::pattern_type;

type NonNullU32 = pattern_type!(u32 is 1..);
type Percent = pattern_type!(u32 is 0..=100);
type Negative = pattern_type!(i32 is ..=0);
type Positive = pattern_type!(i32 is 0..);
type Always = pattern_type!(Option<u32> is Some(_));
//~^ ERROR: pattern not supported in pattern types
