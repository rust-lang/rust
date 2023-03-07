// compile-flags: --extern std=
// error-pattern: extern location for std does not exist
// needs-unwind since it affects the error output
// ignore-emscripten missing eh_catch_typeinfo lang item

fn main() {}
