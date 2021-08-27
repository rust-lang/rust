// revisions: full min
#![cfg_attr(full, feature(const_generics_defaults))]
#![cfg_attr(full, allow(incomplete_features))]

fn bar<const X: u8, 'a>(_: &'a ()) {
    //~^ ERROR lifetime parameters must be declared prior to const parameters
}

fn foo<const X: u8, T>(_: &T) {}
//[min]~^ ERROR type parameters must be declared prior to const parameters

fn main() {}
