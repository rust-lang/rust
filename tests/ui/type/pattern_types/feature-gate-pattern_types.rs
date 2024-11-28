//@ compile-flags: -Zno-analysis

use std::pat::pattern_type;

type NonNullU32 = pattern_type!(u32 is 1..);
//~^ use of unstable library feature `core_pattern_type`
type Percent = pattern_type!(u32 is 0..=100);
//~^ use of unstable library feature `core_pattern_type`
type Negative = pattern_type!(i32 is ..=0);
//~^ use of unstable library feature `core_pattern_type`
type Positive = pattern_type!(i32 is 0..);
//~^ use of unstable library feature `core_pattern_type`
type Always = pattern_type!(Option<u32> is Some(_));
//~^ use of unstable library feature `core_pattern_type`
