// compile-flags: --extern std=
// error-pattern: extern location for std does not exist
// needs-unwind since it affects the error output
// ignore-emscripten compiled with panic=abort, personality not required

fn main() {}
