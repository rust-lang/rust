// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// When denying at the crate level, be sure to not get random warnings from the
// injected intrinsics by the compiler.
#[feature(struct_variant)];
#[deny(missing_doc)];

struct Foo {
    a: int,
    priv b: int,
}

pub struct PubFoo { //~ ERROR: missing documentation
    a: int,      //~ ERROR: missing documentation
    priv b: int,
}

#[allow(missing_doc)]
pub struct PubFoo2 {
    a: int,
    c: int,
}

mod module_no_dox {}
pub mod pub_module_no_dox {} //~ ERROR: missing documentation

/// dox
pub fn foo() {}
pub fn foo2() {} //~ ERROR: missing documentation
fn foo3() {}
#[allow(missing_doc)] pub fn foo4() {}

/// dox
pub trait A {}
trait B {}
pub trait C {} //~ ERROR: missing documentation
#[allow(missing_doc)] pub trait D {}

trait Bar {
    fn foo();
    fn foo_with_impl() {
    }
}

impl Foo {
    pub fn foo() {} //~ ERROR: missing documentation
    /// dox
    pub fn foo1() {}
    fn foo2() {}
    #[allow(missing_doc)] pub fn foo3() {}
}

#[allow(missing_doc)]
trait F {
    fn a();
    fn b(&self);
}

// should need to redefine documentation for implementations of traits
impl F for Foo {
    fn a() {}
    fn b(&self) {}
}

// It sure is nice if doc(hidden) implies allow(missing_doc), and that it
// applies recursively
#[doc(hidden)]
mod a {
    pub fn baz() {}
    pub mod b {
        pub fn baz() {}
    }
}

enum Baz {
    BazA {
        a: int,
        priv b: int
    },
    BarB
}

pub enum PubBaz { //~ ERROR: missing documentation
    PubBazA { //~ ERROR: missing documentation
        a: int, //~ ERROR: missing documentation
        priv b: int
    },

    priv PubBazB
}

/// dox
pub enum PubBaz2 {
    /// dox
    PubBaz2A {
        /// dox
        a: int,
        priv b: int
    },
    priv PubBaz2B
}

#[allow(missing_doc)]
pub enum PubBaz3 {
    PubBaz3A {
        a: int,
        priv b: int
    },
    priv PubBaz3B
}

#[doc(hidden)]
pub fn baz() {}

fn main() {}
