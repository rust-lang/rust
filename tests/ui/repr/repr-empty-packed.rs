//@ compile-flags: --crate-type=lib
#![deny(unused_attributes)]

#[repr()] //~ ERROR unused attribute
#[repr(packed)] //~ ERROR attribute should be applied to a struct or union
pub enum Foo {
    Bar,
    Baz(i32),
}
