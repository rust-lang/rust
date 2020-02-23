// Test the parse error for an overflowing recursion_limit

#![recursion_limit = "999999999999999999999999"]
//~^ ERROR `recursion_limit` must be a non-negative integer
//~| `recursion_limit` is too large

fn main() {}
