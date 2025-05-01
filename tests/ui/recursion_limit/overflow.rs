// Test the parse error for an overflowing recursion_limit

#![recursion_limit = "999999999999999999999999"]
//~^ ERROR `limit` must be a non-negative integer
//~| NOTE `limit` is too large
//~| ERROR `limit` must be a non-negative integer
//~| NOTE `limit` is too large
//~| NOTE duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`

fn main() {}
