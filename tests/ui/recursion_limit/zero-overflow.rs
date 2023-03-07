//~ ERROR overflow evaluating the requirement `&mut Self: DispatchFromDyn<&mut RustaceansAreAwesome>
//~| HELP consider increasing the recursion limit
// build-fail

#![recursion_limit = "0"]

fn main() {}
