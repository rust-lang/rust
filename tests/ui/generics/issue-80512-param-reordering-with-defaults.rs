#![crate_type = "lib"]

struct S<T = (), 'a>(&'a T);
//~^ ERROR lifetime parameters must be declared prior to type and const parameters
