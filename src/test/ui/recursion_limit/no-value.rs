// Test the parse error for no value provided to recursion_limit

#![recursion_limit]
//~^ ERROR malformed `recursion_limit` attribute input

fn main() {}
