#![no_std]
#![crate_type = "lib"]

#[macro_use]
extern crate foo;

#[derive(A)]
enum A<T> {
    Variant(T),
}
