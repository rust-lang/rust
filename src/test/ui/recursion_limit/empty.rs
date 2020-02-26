// Test the parse error for an empty recursion_limit

#![recursion_limit = ""] //~ ERROR `recursion_limit` must be a non-negative integer
                         //~| `recursion_limit` must be a non-negative integer

fn main() {}
