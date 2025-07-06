// Test that when in MIR the amount of local_decls and amount of normalized_input_tys don't match
// that an out-of-bounds access does not occur.
#![feature(c_variadic)]

fn main() {}

fn foo(_: Bar, ...) -> impl {}
//~^ ERROR: `...` is not supported for non-extern functions
//~| ERROR functions with a C variable argument list must be unsafe
//~| ERROR cannot find type `Bar` in this scope
//~| ERROR at least one trait must be specified
