//~ ERROR queries overflow the depth limit!
//~| HELP consider increasing the recursion limit
//@ build-fail

#![recursion_limit = "0"]

fn main() {}
