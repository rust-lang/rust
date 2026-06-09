// Regression test for #121443
// Checks that no ICE occurs upon encountering
// a tuple with unsized element that is not
// the last element

type Fn = dyn FnOnce() -> u8;

const TEST: Fn = some_fn;
//~^ ERROR cannot find value `some_fn` in this scope
//~| ERROR the size for values of type `(dyn FnOnce() -> u8 + 'static)` cannot be known at compilation time
const TEST2: (Fn, u8) = (TEST, 0);
//~^ ERROR the size for values of type `(dyn FnOnce() -> u8 + 'static)` cannot be known at compilation time
//~| ERROR the size for values of type `(dyn FnOnce() -> u8 + 'static)` cannot be known at compilation time

fn main() {}
