//~ ERROR extern location for std does not exist
//@ compile-flags: --extern std=
//@ needs-unwind since it affects the error output
//@ ignore-emscripten missing eh_catch_typeinfo lang item

fn main() {}

//~? ERROR `#[panic_handler]` function required, but not found
//~? ERROR unwinding panics are not supported without std
