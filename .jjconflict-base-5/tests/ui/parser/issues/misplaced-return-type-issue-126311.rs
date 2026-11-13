fn foo<T>() where T: Default -> u8 {}
//~^ ERROR return type should be specified after the function parameters
//~| HELP place the return type after the function parameters

fn main() {}
