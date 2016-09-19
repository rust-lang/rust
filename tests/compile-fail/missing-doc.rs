/* This file incorporates work covered by the following copyright and
 * permission notice:
 *   Copyright 2013 The Rust Project Developers. See the COPYRIGHT
 *   file at the top-level directory of this distribution and at
 *   http://rust-lang.org/COPYRIGHT.
 *
 *   Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
 *   http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
 *   <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
 *   option. This file may not be copied, modified, or distributed
 *   except according to those terms.
 */

#![feature(plugin)]
#![plugin(clippy)]
#![deny(missing_docs_in_private_items)]

// When denying at the crate level, be sure to not get random warnings from the
// injected intrinsics by the compiler.
#![allow(dead_code)]
#![feature(associated_type_defaults)]

//! Some garbage docs for the crate here
#![doc="More garbage"]

type Typedef = String; //~ ERROR: missing documentation for a type alias
pub type PubTypedef = String; //~ ERROR: missing documentation for a type alias

struct Foo { //~ ERROR: missing documentation for a struct
    a: isize, //~ ERROR: missing documentation for a struct field
    b: isize, //~ ERROR: missing documentation for a struct field
}

pub struct PubFoo { //~ ERROR: missing documentation for a struct
    pub a: isize,      //~ ERROR: missing documentation for a struct field
    b: isize, //~ ERROR: missing documentation for a struct field
}

#[allow(missing_docs_in_private_items)]
pub struct PubFoo2 {
    pub a: isize,
    pub c: isize,
}

mod module_no_dox {} //~ ERROR: missing documentation for a module
pub mod pub_module_no_dox {} //~ ERROR: missing documentation for a module

/// dox
pub fn foo() {}
pub fn foo2() {} //~ ERROR: missing documentation for a function
fn foo3() {} //~ ERROR: missing documentation for a function
#[allow(missing_docs_in_private_items)] pub fn foo4() {}

/// dox
pub trait A {
    /// dox
    fn foo(&self);
    /// dox
    fn foo_with_impl(&self) {}
}

#[allow(missing_docs_in_private_items)]
trait B {
    fn foo(&self);
    fn foo_with_impl(&self) {}
}

pub trait C { //~ ERROR: missing documentation for a trait
    fn foo(&self); //~ ERROR: missing documentation for a trait method
    fn foo_with_impl(&self) {} //~ ERROR: missing documentation for a trait method
}

#[allow(missing_docs_in_private_items)]
pub trait D {
    fn dummy(&self) { }
}

/// dox
pub trait E {
    type AssociatedType; //~ ERROR: missing documentation for an associated type
    type AssociatedTypeDef = Self; //~ ERROR: missing documentation for an associated type

    /// dox
    type DocumentedType;
    /// dox
    type DocumentedTypeDef = Self;
    /// dox
    fn dummy(&self) {}
}

impl Foo {
    pub fn foo() {} //~ ERROR: missing documentation for a method
    fn bar() {} //~ ERROR: missing documentation for a method
}

impl PubFoo {
    pub fn foo() {} //~ ERROR: missing documentation for a method
    /// dox
    pub fn foo1() {}
    fn foo2() {} //~ ERROR: missing documentation for a method
    #[allow(missing_docs_in_private_items)] pub fn foo3() {}
}

#[allow(missing_docs_in_private_items)]
trait F {
    fn a();
    fn b(&self);
}

// should need to redefine documentation for implementations of traits
impl F for Foo {
    fn a() {}
    fn b(&self) {}
}

// It sure is nice if doc(hidden) implies allow(missing_docs), and that it
// applies recursively
#[doc(hidden)]
mod a {
    pub fn baz() {}
    pub mod b {
        pub fn baz() {}
    }
}

enum Baz { //~ ERROR: missing documentation for an enum
    BazA { //~ ERROR: missing documentation for a variant
        a: isize, //~ ERROR: missing documentation for a struct field
        b: isize //~ ERROR: missing documentation for a struct field
    },
    BarB //~ ERROR: missing documentation for a variant
}

pub enum PubBaz { //~ ERROR: missing documentation for an enum
    PubBazA { //~ ERROR: missing documentation for a variant
        a: isize, //~ ERROR: missing documentation for a struct field
    },
}

/// dox
pub enum PubBaz2 {
    /// dox
    PubBaz2A {
        /// dox
        a: isize,
    },
}

#[allow(missing_docs_in_private_items)]
pub enum PubBaz3 {
    PubBaz3A {
        b: isize
    },
}

#[doc(hidden)]
pub fn baz() {}


const FOO: u32 = 0; //~ ERROR: missing documentation for a const
/// dox
pub const FOO1: u32 = 0;
#[allow(missing_docs_in_private_items)]
pub const FOO2: u32 = 0;
#[doc(hidden)]
pub const FOO3: u32 = 0;
pub const FOO4: u32 = 0; //~ ERROR: missing documentation for a const


static BAR: u32 = 0; //~ ERROR: missing documentation for a static
/// dox
pub static BAR1: u32 = 0;
#[allow(missing_docs_in_private_items)]
pub static BAR2: u32 = 0;
#[doc(hidden)]
pub static BAR3: u32 = 0;
pub static BAR4: u32 = 0; //~ ERROR: missing documentation for a static


mod internal_impl { //~ ERROR: missing documentation for a module
    /// dox
    pub fn documented() {}
    pub fn undocumented1() {} //~ ERROR: missing documentation for a function
    pub fn undocumented2() {} //~ ERROR: missing documentation for a function
    fn undocumented3() {} //~ ERROR: missing documentation for a function
    /// dox
    pub mod globbed {
        /// dox
        pub fn also_documented() {}
        pub fn also_undocumented1() {} //~ ERROR: missing documentation for a function
        fn also_undocumented2() {} //~ ERROR: missing documentation for a function
    }
}
/// dox
pub mod public_interface {
    pub use internal_impl::documented as foo;
    pub use internal_impl::undocumented1 as bar;
    pub use internal_impl::{documented, undocumented2};
    pub use internal_impl::globbed::*;
}

fn main() {} //~ ERROR: missing documentation for a function
