//@needs-asm-support
//@aux-build: proc_macros.rs

#![warn(clippy::missing_docs_in_private_items)]
#![allow(dead_code, non_local_definitions)]

extern crate proc_macros;

use proc_macros::with_span;
use std::arch::global_asm;

type Typedef = String;
//~^ missing_docs_in_private_items
pub type PubTypedef = String;

mod module_no_dox {}
//~^ missing_docs_in_private_items
pub mod pub_module_no_dox {}

/// dox
pub fn foo() {}
pub fn foo2() {}
fn foo3() {}
//~^ missing_docs_in_private_items
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
    //~^ missing_docs_in_private_items
    BazA { a: isize, b: isize },
    //~^ missing_docs_in_private_items
    //~| missing_docs_in_private_items
    //~| missing_docs_in_private_items
    BarB,
    //~^ missing_docs_in_private_items
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
//~^ missing_docs_in_private_items
/// dox
pub const FOO1: u32 = 0;
#[allow(clippy::missing_docs_in_private_items)]
pub const FOO2: u32 = 0;
#[doc(hidden)]
pub const FOO3: u32 = 0;
pub const FOO4: u32 = 0;

static BAR: u32 = 0;
//~^ missing_docs_in_private_items
/// dox
pub static BAR1: u32 = 0;
#[allow(clippy::missing_docs_in_private_items)]
pub static BAR2: u32 = 0;
#[doc(hidden)]
pub static BAR3: u32 = 0;
pub static BAR4: u32 = 0;

mod internal_impl {
    //~^ missing_docs_in_private_items
    /// dox
    pub fn documented() {}
    pub fn undocumented1() {}
    pub fn undocumented2() {}
    fn undocumented3() {}
    //~^ missing_docs_in_private_items
    /// dox
    pub mod globbed {
        /// dox
        pub fn also_documented() {}
        pub fn also_undocumented1() {}
        fn also_undocumented2() {}
        //~^ missing_docs_in_private_items
    }
}
/// dox
pub mod public_interface {
    pub use crate::internal_impl::globbed::*;
    pub use crate::internal_impl::{documented as foo, documented, undocumented1 as bar, undocumented2};
}

fn main() {}

// Ensure global asm doesn't require documentation.
global_asm! { "" }

// Don't lint proc macro output with an unexpected span.
with_span!(span pub struct FooPm { pub field: u32});
with_span!(span pub struct FooPm2;);
with_span!(span pub enum FooPm3 { A, B(u32), C { field: u32 }});
with_span!(span pub fn foo_pm() {});
with_span!(span pub static FOO_PM: u32 = 0;);
with_span!(span pub const FOO2_PM: u32 = 0;);

// Don't lint unnamed constants
const _: () = ();

fn issue13298() {
    //~^ missing_docs_in_private_items
    // Rustdoc doesn't generate documentation for items within other items like fns or consts
    const MSG: &str = "Hello, world!";
}

with_span!(span
    /// docs
    struct WithInternalHelper {
        __helper: u32,
        $(
            field: u32, //~ missing_docs_in_private_items
            /// docs
            field2: u32,
        )
    }

    const _: () = {
        struct Internal {
            __helper: u32,
            $(field: u32),
            $(field2: u32),
        };
        impl WithInternalHelper {
            pub(crate) fn internal(self) -> Internal {
                unimplemented!()
            }
        }
    };
);

/// docs
struct WithHiddenImpl;

/// docs
fn with_hidden_impl() {
    mod m {
        struct Bar;
        const _: () = {
            mod m2 {
                impl super::Bar {
                    pub fn bar() {
                        impl crate::WithHiddenImpl {
                            pub(crate) fn visible(&self) {}
                        }
                    }
                }
            }

            #[automatically_derived]
            impl crate::WithHiddenImpl {
                fn derived(self) {}
            }
        };
    }
}
