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

// Do not remove on snapshot creation. Needed for bootstrap. (Issue #22364)
#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "syntax"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![cfg_attr(stage0, staged_api)]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       test(attr(deny(warnings))))]

#![cfg_attr(stage0, feature(rustc_attrs))]
#![cfg_attr(stage0, allow(unused_attributes))]
#![feature(associated_consts)]
#![feature(filling_drop)]
#![feature(libc)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(str_char)]
#![feature(str_escape)]
#![feature(unicode)]

extern crate fmt_macros;
extern crate serialize;
extern crate term;
extern crate libc;
#[macro_use] extern crate log;
#[macro_use] #[no_link] extern crate rustc_bitflags;

extern crate serialize as rustc_serialize; // used by deriving

// A variant of 'try!' that panics on Err(FatalError). This is used as a
// crutch on the way towards a non-panic!-prone parser. It should be used
// for fatal parsing errors; eventually we plan to convert all code using
// panictry to just use normal try
macro_rules! panictry {
    ($e:expr) => ({
        use std::result::Result::{Ok, Err};
        use diagnostic::FatalError;
        match $e {
            Ok(e) => e,
            Err(FatalError) => panic!(FatalError)
        }
    })
}

pub mod util {
    pub mod interner;
    pub mod lev_distance;
    pub mod node_count;
    pub mod parser;
    #[cfg(test)]
    pub mod parser_testing;
    pub mod small_vector;
    pub mod move_map;
}

pub mod diagnostics {
    pub mod macros;
    pub mod plugin;
    pub mod registry;
    pub mod metadata;
}

pub mod syntax {
    pub use ext;
    pub use parse;
    pub use ast;
}

pub mod abi;
pub mod ast;
pub mod ast_util;
pub mod attr;
pub mod codemap;
pub mod config;
pub mod diagnostic;
pub mod entry;
pub mod feature_gate;
pub mod fold;
pub mod owned_slice;
pub mod parse;
pub mod ptr;
pub mod show_span;
pub mod std_inject;
pub mod str;
pub mod test;
pub mod visit;

pub mod print {
    pub mod pp;
    pub mod pprust;
}

pub mod ext {
    pub mod asm;
    pub mod base;
    pub mod build;
    pub mod cfg;
    pub mod concat;
    pub mod concat_idents;
    pub mod deriving;
    pub mod env;
    pub mod expand;
    pub mod format;
    pub mod log_syntax;
    pub mod mtwt;
    pub mod quote;
    pub mod source_util;
    pub mod trace_macros;

    pub mod tt {
        pub mod transcribe;
        pub mod macro_parser;
        pub mod macro_rules;
    }
}
