//! The Rust parser and macro expander.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       test(attr(deny(warnings))))]

#![deny(rust_2018_idioms)]

#![feature(crate_visibility_modifier)]
#![feature(label_break_value)]
#![feature(rustc_attrs)]
#![feature(rustc_diagnostic_macros)]
#![feature(slice_sort_by_cached_key)]
#![feature(str_escape)]
#![feature(step_trait)]
#![feature(try_trait)]
#![feature(unicode_internals)]

#![recursion_limit="256"]

#[allow(unused_extern_crates)]
extern crate serialize as rustc_serialize; // used by deriving

pub use rustc_errors as errors;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::bit_set::GrowableBitSet;
pub use rustc_data_structures::thin_vec::ThinVec;
use ast::AttrId;

// A variant of 'try!' that panics on an Err. This is used as a crutch on the
// way towards a non-panic!-prone parser. It should be used for fatal parsing
// errors; eventually we plan to convert all code using panictry to just use
// normal try.
macro_rules! panictry {
    ($e:expr) => ({
        use std::result::Result::{Ok, Err};
        use crate::errors::FatalError;
        match $e {
            Ok(e) => e,
            Err(mut e) => {
                e.emit();
                FatalError.raise()
            }
        }
    })
}

// A variant of 'panictry!' that works on a Vec<Diagnostic> instead of a single DiagnosticBuilder.
macro_rules! panictry_buffer {
    ($handler:expr, $e:expr) => ({
        use std::result::Result::{Ok, Err};
        use crate::errors::{FatalError, DiagnosticBuilder};
        match $e {
            Ok(e) => e,
            Err(errs) => {
                for e in errs {
                    DiagnosticBuilder::new_diagnostic($handler, e).emit();
                }
                FatalError.raise()
            }
        }
    })
}

#[macro_export]
macro_rules! unwrap_or {
    ($opt:expr, $default:expr) => {
        match $opt {
            Some(x) => x,
            None => $default,
        }
    }
}

pub struct Globals {
    used_attrs: Lock<GrowableBitSet<AttrId>>,
    known_attrs: Lock<GrowableBitSet<AttrId>>,
    syntax_pos_globals: syntax_pos::Globals,
}

impl Globals {
    fn new() -> Globals {
        Globals {
            // We have no idea how many attributes their will be, so just
            // initiate the vectors with 0 bits. We'll grow them as necessary.
            used_attrs: Lock::new(GrowableBitSet::new_empty()),
            known_attrs: Lock::new(GrowableBitSet::new_empty()),
            syntax_pos_globals: syntax_pos::Globals::new(),
        }
    }
}

pub fn with_globals<F, R>(f: F) -> R
    where F: FnOnce() -> R
{
    let globals = Globals::new();
    GLOBALS.set(&globals, || {
        syntax_pos::GLOBALS.set(&globals.syntax_pos_globals, f)
    })
}

scoped_tls::scoped_thread_local!(pub static GLOBALS: Globals);

#[macro_use]
pub mod diagnostics {
    #[macro_use]
    pub mod macros;
    pub mod plugin;
    pub mod metadata;
}

// N.B., this module needs to be declared first so diagnostics are
// registered before they are used.
pub mod diagnostic_list;

pub mod util {
    pub mod lev_distance;
    pub mod node_count;
    pub mod parser;
    #[cfg(test)]
    pub mod parser_testing;
    pub mod map_in_place;
}

pub mod json;

pub mod syntax {
    pub use crate::ext;
    pub use crate::parse;
    pub use crate::ast;
}

pub mod ast;
pub mod attr;
pub mod source_map;
#[macro_use]
pub mod config;
pub mod entry;
pub mod feature_gate;
pub mod mut_visit;
pub mod parse;
pub mod ptr;
pub mod show_span;
pub mod std_inject;
pub use syntax_pos::edition;
pub use syntax_pos::symbol;
pub mod test;
pub mod tokenstream;
pub mod visit;

pub mod print {
    pub mod pp;
    pub mod pprust;
}

pub mod ext {
    pub use syntax_pos::hygiene;
    pub mod base;
    pub mod build;
    pub mod derive;
    pub mod expand;
    pub mod placeholders;
    pub mod source_util;

    pub mod tt {
        pub mod transcribe;
        pub mod macro_parser;
        pub mod macro_rules;
        pub mod quoted;
    }
}

pub mod early_buffered_lints;

#[cfg(test)]
mod test_snippet;

__build_diagnostic_array! { libsyntax, DIAGNOSTICS }
