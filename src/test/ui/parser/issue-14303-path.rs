// compile-flags: -Z parse-only

fn bar<'a, T>(x: mymodule::X<'a, T, 'b, 'c>) {}
//~^ ERROR lifetime parameters must be declared prior to type parameters
