//~ ERROR extern location for std does not exist
//~^ ERROR cannot resolve a prelude import
//@ compile-flags: --extern std=
//@ needs-unwind since it affects the error output

fn main() {}

//~? ERROR `#[panic_handler]` function required, but not found
//~? ERROR unwinding panics are not supported without std
