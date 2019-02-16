#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn foo<const X: (), T>(_: T) {
    //~^ ERROR type parameters must be declared prior to const parameters
    //~^^ ERROR const generics in any position are currently unsupported
}

fn bar<const X: (), 'a>(_: &'a ()) {
    //~^ ERROR lifetime parameters must be declared prior to const parameters
}

fn main() {}
