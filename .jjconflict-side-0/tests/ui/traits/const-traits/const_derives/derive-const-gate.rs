#[derive_const(Debug)] //~ ERROR use of unstable library feature
//~^ ERROR const `impl` for trait `Debug` which is not `const`
//~| ERROR cannot call non-const method
pub struct S;

fn main() {}
