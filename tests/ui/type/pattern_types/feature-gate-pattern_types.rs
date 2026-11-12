//@ revisions: disabled enabled
//@ compile-flags: -Zno-analysis
#![cfg_attr(enabled, feature(pattern_types))]

use std::pat::pattern_type;

type NonNullU32 = pattern_type!(u32 is 1..);
//[disabled]~^ ERROR use of unstable library feature `pattern_types`
type Percent = pattern_type!(u32 is 0..=100);
//[disabled]~^ ERROR use of unstable library feature `pattern_types`
type Negative = pattern_type!(i32 is ..=0);
//[disabled]~^ ERROR use of unstable library feature `pattern_types`
type Positive = pattern_type!(i32 is 0..);
//[disabled]~^ ERROR use of unstable library feature `pattern_types`
type Always = pattern_type!(Option<u32> is Some(_));
//[disabled]~^ ERROR use of unstable library feature `pattern_types`
//~^^ ERROR pattern not supported in pattern types
