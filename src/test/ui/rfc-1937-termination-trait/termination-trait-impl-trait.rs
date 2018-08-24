// Tests that an `impl Trait` that is not `impl Termination` will not work.
fn main() -> impl Copy { }
//~^ ERROR `main` has invalid return type `impl std::marker::Copy`
