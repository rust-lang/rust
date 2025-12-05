#![feature(decl_macro)]

pub macro addr_of($place:expr) {
    &raw const $place
}

pub macro addr_of_crate($place:expr) {
    &raw const $place
}

pub macro addr_of_super($place:expr) {
    &raw const $place
}

pub macro addr_of_self($place:expr) {
    &raw const $place
}

pub macro addr_of_local($place:expr) {
    &raw const $place
}

pub struct Foo;
pub struct FooCrate;
pub struct FooSuper;
pub struct FooSelf;
pub struct FooLocal;

pub enum Bar { Foo, }
pub enum BarCrate { Foo, }
pub enum BarSuper { Foo, }
pub enum BarSelf { Foo, }
pub enum BarLocal { Foo, }

pub fn foo() {}
pub fn foo_crate() {}
pub fn foo_super() {}
pub fn foo_self() {}
pub fn foo_local() {}

pub type Type = i32;
pub type TypeCrate = i32;
pub type TypeSuper = i32;
pub type TypeSelf = i32;
pub type TypeLocal = i32;

pub union Union {
    a: i8,
    b: i8,
}
pub union UnionCrate {
    a: i8,
    b: i8,
}
pub union UnionSuper {
    a: i8,
    b: i8,
}
pub union UnionSelf {
    a: i8,
    b: i8,
}
pub union UnionLocal {
    a: i8,
    b: i8,
}
