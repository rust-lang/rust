// Test the parse error for an invalid digit in recursion_limit

#![recursion_limit = "-100"] //~ ERROR `recursion_limit` must be a non-negative integer
                             //~| not a valid integer

fn main() {}
