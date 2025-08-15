//~ ERROR overflow evaluating the requirement
//~| HELP consider increasing the recursion limit
//@ build-fail

#![recursion_limit = "0"]

fn main() {}
