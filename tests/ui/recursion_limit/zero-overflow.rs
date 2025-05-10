//~ ERROR overflow evaluating the requirement `&mut Self: DispatchFromDyn<&mut RustaceansAreAwesome>
//~| HELP consider increasing the recursion limit by adding a `#![recursion_limit = "2"]` attribute to your crate (`zero_overflow`)
//~^^ ERROR queries overflow the depth limit!
//~| HELP consider increasing the recursion limit by adding a `#![recursion_limit = "2"]` attribute to your crate (`zero_overflow`)
//@ build-fail

#![recursion_limit = "0"]

fn main() {}
