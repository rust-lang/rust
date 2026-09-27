//@ proc-macro: derive-no-generics.rs
#![crate_type = "lib"]

#[macro_use]
extern crate derive_no_generics;

#[derive(A)]
enum A<T> {
    //~^ ERROR: missing generics for enum `A`
    //~| ERROR: missing generics for enum `A`
    Variant(T),
    //~^ ERROR: cannot find type `T` in this scope
}
