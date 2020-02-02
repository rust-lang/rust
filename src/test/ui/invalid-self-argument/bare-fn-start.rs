fn a(&self) { }
//~^ ERROR `self` parameter is only allowed in associated functions
//~| NOTE not semantically valid as function parameter
//~| NOTE associated functions are those in `impl` or `trait` definitions

fn main() { }
