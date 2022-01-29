// aux-build:issue-57271-lib.rs

extern crate issue_57271_lib;

use issue_57271_lib::BaseType;

pub enum ObjectType { //~ ERROR recursive type `ObjectType` has infinite size
    Class(ClassTypeSignature),
    Array(TypeSignature),
    TypeVariable(()),
}

pub struct ClassTypeSignature {
    pub package: (),
    pub class: (),
    pub inner: (),
}

pub enum TypeSignature { //~ ERROR recursive type `TypeSignature` has infinite size
    Base(BaseType),
    Object(ObjectType),
}

fn main() {}
