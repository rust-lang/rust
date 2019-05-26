fn b(foo: u32, &mut self) { }
//~^ ERROR unexpected `self` parameter in function
//~| NOTE not valid as function parameter
//~| NOTE `self` is only valid as the first parameter of an associated function

fn main() { }
