//@aux-build: proc_macros.rs

#![warn(clippy::missing_docs_in_private_items)]
#![allow(dead_code)]
#![feature(associated_type_defaults)]

//! Some garbage docs for the crate here
#![doc = "More garbage"]

extern crate proc_macros;
use proc_macros::with_span;

struct Foo {
    //~^ missing_docs_in_private_items
    a: isize,
    //~^ missing_docs_in_private_items
    b: isize,
    //~^ missing_docs_in_private_items
}

pub struct PubFoo {
    pub a: isize,
    b: isize,
    //~^ missing_docs_in_private_items
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
    pub fn new() -> Self {
        //~^ missing_docs_in_private_items
        Foo { a: 0, b: 0 }
    }
    fn bar() {}
    //~^ missing_docs_in_private_items
}

impl PubFoo {
    pub fn foo() {}
    /// dox
    pub fn foo1() {}
    #[must_use = "yep"]
    fn foo2() -> u32 {
        //~^ missing_docs_in_private_items
        1
    }
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

// don't lint proc macro output
with_span!(span
    pub struct FooPm;
    impl FooPm {
        pub fn foo() {}
        pub const fn bar() {}
        pub const X: u32 = 0;
    }
);
