#![warn(clippy::missing_docs_in_private_items)]
#![allow(dead_code)]
#![feature(associated_type_defaults)]

//! Some garbage docs for the crate here
#![doc = "More garbage"]

struct Foo {
    a: isize,
    b: isize,
}

pub struct PubFoo {
    pub a: isize,
    b: isize,
}

#[allow(clippy::missing_docs_in_private_items)]
pub struct PubFoo2 {
    pub a: isize,
    pub c: isize,
}

/// dox
pub trait A {
    /// dox
    fn foo(&self);
    /// dox
    fn foo_with_impl(&self) {}
}

#[allow(clippy::missing_docs_in_private_items)]
trait B {
    fn foo(&self);
    fn foo_with_impl(&self) {}
}

pub trait C {
    fn foo(&self);
    fn foo_with_impl(&self) {}
}

#[allow(clippy::missing_docs_in_private_items)]
pub trait D {
    fn dummy(&self) {}
}

/// dox
pub trait E: Sized {
    type AssociatedType;
    type AssociatedTypeDef = Self;

    /// dox
    type DocumentedType;
    /// dox
    type DocumentedTypeDef = Self;
    /// dox
    fn dummy(&self) {}
}

impl Foo {
    pub fn foo() {}
    fn bar() {}
}

impl PubFoo {
    pub fn foo() {}
    /// dox
    pub fn foo1() {}
    fn foo2() {}
    #[allow(clippy::missing_docs_in_private_items)]
    pub fn foo3() {}
}

#[allow(clippy::missing_docs_in_private_items)]
trait F {
    fn a();
    fn b(&self);
}

// should need to redefine documentation for implementations of traits
impl F for Foo {
    fn a() {}
    fn b(&self) {}
}

fn main() {}
