fn bar<'a, T>(x: mymodule::X<'a, T, 'b, 'c>) {}
//~^ ERROR lifetime parameters must be declared prior to type parameters
//~| ERROR failed to resolve: use of undeclared type or module `mymodule`
//~| ERROR use of undeclared lifetime name `'b`
//~| ERROR use of undeclared lifetime name `'c`

fn main() {}
