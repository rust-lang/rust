// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The Rust parser and macro expander.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![crate_name = "syntax"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       test(attr(deny(warnings))))]
#![deny(warnings)]

#![feature(associated_consts)]
#![feature(const_fn)]
#![feature(optin_builtin_traits)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(str_escape)]
#![feature(unicode)]
#![feature(rustc_diagnostic_macros)]
#![feature(specialization)]

extern crate serialize;
#[macro_use] extern crate log;
#[macro_use] #[no_link] extern crate rustc_bitflags;
extern crate std_unicode;
pub extern crate rustc_errors as errors;
extern crate syntax_pos;
extern crate rustc_data_structures;

extern crate rustc_i128;

extern crate serialize as rustc_serialize; // used by deriving

// A variant of 'try!' that panics on an Err. This is used as a crutch on the
// way towards a non-panic!-prone parser. It should be used for fatal parsing
// errors; eventually we plan to convert all code using panictry to just use
// normal try.
// Exported for syntax_ext, not meant for general use.
#[macro_export]
macro_rules! panictry {
    ($e:expr) => ({
        use std::result::Result::{Ok, Err};
        use errors::FatalError;
        match $e {
            Ok(e) => e,
            Err(mut e) => {
                e.emit();
                panic!(FatalError);
            }
        }
    })
}

#[macro_use]
pub mod diagnostics {
    #[macro_use]
    pub mod macros;
    pub mod plugin;
    pub mod metadata;
}

// NB: This module needs to be declared first so diagnostics are
// registered before they are used.
pub mod diagnostic_list;

pub mod util {
    pub mod lev_distance;
    pub mod node_count;
    pub mod parser;
    #[cfg(test)]
    pub mod parser_testing;
    pub mod small_vector;
    pub mod move_map;

    mod thin_vec;
    pub use self::thin_vec::ThinVec;
}

pub mod json;

pub mod syntax {
    pub use ext;
    pub use parse;
    pub use ast;
}

pub mod abi;
pub mod ast;
pub mod attr;
pub mod codemap;
#[macro_use]
pub mod config;
pub mod entry;
pub mod feature_gate;
pub mod fold;
pub mod parse;
pub mod ptr;
pub mod show_span;
pub mod std_inject;
pub mod str;
pub mod symbol;
pub mod test;
pub mod tokenstream;
pub mod visit;

pub mod print {
    pub mod pp;
    pub mod pprust;
}

pub mod ext {
    pub mod base;
    pub mod build;
    pub mod expand;
    pub mod placeholders;
    pub mod hygiene;
    pub mod proc_macro_shim;
    pub mod quote;
    pub mod source_util;

    pub mod tt {
        pub mod transcribe;
        pub mod macro_parser;
        pub mod macro_rules;
    }
}

#[cfg(test)]
mod test_snippet;

// __build_diagnostic_array! { libsyntax, DIAGNOSTICS }
