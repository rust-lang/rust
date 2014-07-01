// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The Rust parser and macro expander.

# Note

This API is completely unstable and subject to change.

*/

#![crate_id = "syntax#0.11.0"] // NOTE: remove after stage0
#![crate_name = "syntax"]
#![experimental]
#![license = "MIT/ASL2"]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/0.11.0/")]

#![feature(macro_rules, globs, managed_boxes, default_type_params, phase)]
#![feature(quote, unsafe_destructor)]
#![allow(deprecated)]
#![allow(unused_attribute)] // NOTE: remove after stage0

extern crate serialize;
extern crate term;
#[phase(plugin, link)] extern crate log;

extern crate fmt_macros;
extern crate debug;

pub mod util {
    pub mod interner;
    #[cfg(test)]
    pub mod parser_testing;
    pub mod small_vector;
}

pub mod syntax {
    pub use ext;
    pub use parse;
    pub use ast;
}

pub mod owned_slice;
pub mod attr;
pub mod diagnostic;
pub mod codemap;
pub mod abi;
pub mod ast;
pub mod ast_util;
pub mod ast_map;
pub mod visit;
pub mod fold;


pub mod parse;
pub mod crateid;

pub mod print {
    pub mod pp;
    pub mod pprust;
}

pub mod ext {
    pub mod asm;
    pub mod base;
    pub mod expand;

    pub mod quote;

    pub mod deriving;

    pub mod build;

    pub mod tt {
        pub mod transcribe;
        pub mod macro_parser;
        pub mod macro_rules;
    }

    pub mod mtwt;

    pub mod cfg;
    pub mod fmt;
    pub mod format;
    pub mod env;
    pub mod bytes;
    pub mod concat;
    pub mod concat_idents;
    pub mod log_syntax;
    pub mod source_util;

    pub mod trace_macros;
}
