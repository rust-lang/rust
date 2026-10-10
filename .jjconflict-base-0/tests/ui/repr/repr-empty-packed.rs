//@ compile-flags: --crate-type=lib
#![deny(unused_attributes)]

#[repr()] //~ ERROR unused attribute
#[repr(packed)] //~ ERROR attribute cannot be used on enums
pub enum Foo {
    Bar,
    Baz(i32),
}
