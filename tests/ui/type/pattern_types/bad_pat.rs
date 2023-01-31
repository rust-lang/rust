#![feature(pattern_types)]

type NonNullU32_2 = u32 is 1..=;
//~^ ERROR: inclusive range with no end
type Positive2 = i32 is 0..=;
//~^ ERROR: inclusive range with no end
type Wild = () is _;
//~^ ERROR: wildcard patterns are not permitted for pattern types

fn main() {}
