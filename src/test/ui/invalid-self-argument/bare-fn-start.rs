fn a(&self) { }
//~^ ERROR `self` parameter only allowed in associated `fn`s
//~| NOTE not semantically valid as function parameter
//~| NOTE associated `fn`s are those in `impl` or `trait` definitions

fn main() { }
