#![feature(decl_macro)]

pub macro addr_of($place:expr) {
    &raw const $place
}

pub macro addr_of_self($place:expr) {
    &raw const $place
}

pub macro addr_of_crate($place:expr) {
    &raw const $place
}

pub struct Foo;
pub struct FooSelf;
pub struct FooCrate;

pub enum Bar { Foo, }
pub enum BarSelf { Foo, }
pub enum BarCrate { Foo, }

pub fn foo() {}
pub fn foo_self() {}
pub fn foo_crate() {}

pub type Type = i32;
pub type TypeSelf = i32;
pub type TypeCrate = i32;

pub union Union {
    a: i8,
    b: i8,
}
pub union UnionSelf {
    a: i8,
    b: i8,
}
pub union UnionCrate {
    a: i8,
    b: i8,
}
