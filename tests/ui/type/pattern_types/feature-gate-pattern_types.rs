//@ compile-flags: -Zno-analysis

use std::pat::pattern_type;

type NonNullU32 = pattern_type!(u32 is 1..);
//~^ use of unstable library feature `pattern_type_macro`
type Percent = pattern_type!(u32 is 0..=100);
//~^ use of unstable library feature `pattern_type_macro`
type Negative = pattern_type!(i32 is ..=0);
//~^ use of unstable library feature `pattern_type_macro`
type Positive = pattern_type!(i32 is 0..);
//~^ use of unstable library feature `pattern_type_macro`
type Always = pattern_type!(Option<u32> is Some(_));
//~^ use of unstable library feature `pattern_type_macro`
//~| ERROR pattern not supported in pattern types
