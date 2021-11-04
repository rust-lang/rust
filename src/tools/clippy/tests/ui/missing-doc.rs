#![warn(clippy::missing_docs_in_private_items)]
// When denying at the crate level, be sure to not get random warnings from the
// injected intrinsics by the compiler.
#![allow(dead_code)]
#![feature(global_asm)]
//! Some garbage docs for the crate here
#![doc = "More garbage"]

type Typedef = String;
pub type PubTypedef = String;

mod module_no_dox {}
pub mod pub_module_no_dox {}

/// dox
pub fn foo() {}
pub fn foo2() {}
fn foo3() {}
#[allow(clippy::missing_docs_in_private_items)]
pub fn foo4() {}

// It sure is nice if doc(hidden) implies allow(missing_docs), and that it
// applies recursively
#[doc(hidden)]
mod a {
    pub fn baz() {}
    pub mod b {
        pub fn baz() {}
    }
}

enum Baz {
    BazA { a: isize, b: isize },
    BarB,
}

pub enum PubBaz {
    PubBazA { a: isize },
}

/// dox
pub enum PubBaz2 {
    /// dox
    PubBaz2A {
        /// dox
        a: isize,
    },
}

#[allow(clippy::missing_docs_in_private_items)]
pub enum PubBaz3 {
    PubBaz3A { b: isize },
}

#[doc(hidden)]
pub fn baz() {}

const FOO: u32 = 0;
/// dox
pub const FOO1: u32 = 0;
#[allow(clippy::missing_docs_in_private_items)]
pub const FOO2: u32 = 0;
#[doc(hidden)]
pub const FOO3: u32 = 0;
pub const FOO4: u32 = 0;

static BAR: u32 = 0;
/// dox
pub static BAR1: u32 = 0;
#[allow(clippy::missing_docs_in_private_items)]
pub static BAR2: u32 = 0;
#[doc(hidden)]
pub static BAR3: u32 = 0;
pub static BAR4: u32 = 0;

mod internal_impl {
    /// dox
    pub fn documented() {}
    pub fn undocumented1() {}
    pub fn undocumented2() {}
    fn undocumented3() {}
    /// dox
    pub mod globbed {
        /// dox
        pub fn also_documented() {}
        pub fn also_undocumented1() {}
        fn also_undocumented2() {}
    }
}
/// dox
pub mod public_interface {
    pub use crate::internal_impl::documented as foo;
    pub use crate::internal_impl::globbed::*;
    pub use crate::internal_impl::undocumented1 as bar;
    pub use crate::internal_impl::{documented, undocumented2};
}

fn main() {}

// Ensure global asm doesn't require documentation.
global_asm! { "" }
