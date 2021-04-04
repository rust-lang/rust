// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn bar<const X: (), 'a>(_: &'a ()) {
    //~^ ERROR lifetime parameters must be declared prior to const parameters
    //[min]~^^ ERROR `()` is forbidden as the type of a const generic parameter
}

fn foo<const X: (), T>(_: &T) {}
//[min]~^ ERROR type parameters must be declared prior to const parameters
//[min]~^^ ERROR `()` is forbidden as the type of a const generic parameter

fn main() {}
