#![feature(inherent_associated_types)]

pub struct S;

impl S {
    pub type TYPE = ();
    pub const CONST: usize = 0;

    pub fn foo() {}
}

pub trait Trait {
    fn bar() {}
}

impl Trait for S {}

pub struct X<T>(T);

impl X<usize> {
    pub fn foo() {}
}

impl X<String> {
    pub fn foo() {}
}
