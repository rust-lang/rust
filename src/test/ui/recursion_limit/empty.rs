// Test the parse error for an empty recursion_limit

#![recursion_limit = ""] //~ ERROR `limit` must be a non-negative integer
                         //~| `limit` must be a non-negative integer

fn main() {}
