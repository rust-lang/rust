fn b(foo: u32, &mut self) { }
//~^ ERROR unexpected `self` parameter in function
//~| NOTE must be the first parameter of an associated function

fn main() { }
