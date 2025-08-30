// https://github.com/rust-lang/rust/issues/57271
//@ aux-build:aux-57271-lib.rs

extern crate aux_57271_lib;

use aux_57271_lib::BaseType;

pub enum ObjectType { //~ ERROR recursive types `ObjectType` and `TypeSignature` have infinite size
    Class(ClassTypeSignature),
    Array(TypeSignature),
    TypeVariable(()),
}

pub struct ClassTypeSignature {
    pub package: (),
    pub class: (),
    pub inner: (),
}

pub enum TypeSignature {
    Base(BaseType),
    Object(ObjectType),
}

fn main() {}
