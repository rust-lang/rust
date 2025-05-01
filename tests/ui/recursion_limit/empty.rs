// Test the parse error for an empty recursion_limit

#![recursion_limit = ""] //~ ERROR `limit` must be a non-negative integer
                         //~| NOTE `limit` must be a non-negative integer
                         //~| ERROR `limit` must be a non-negative integer
                         //~| NOTE `limit` must be a non-negative integer
                         //~| NOTE duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`

fn main() {}
