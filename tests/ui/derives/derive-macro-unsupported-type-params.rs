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

// Resolution error comes entirely from within the derive macro:
#[derive(B)] //~ ERROR: cannot find type `T` in this scope
enum B<T> {
    //~^ ERROR: missing generics for enum `B`
    //~| ERROR: missing generics for enum `B`
    Variant(T),
}

#[derive(C)]
enum C<T> {
    Variant(T),
    //~^ ERROR: cannot find value `T` in this scope
    //~| ERROR: cannot find type `T` in this scope
}
