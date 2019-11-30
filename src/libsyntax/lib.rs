//! The Rust parser and macro expander.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/",
       test(attr(deny(warnings))))]

#![feature(box_syntax)]
#![feature(const_fn)]
#![feature(const_transmute)]
#![feature(crate_visibility_modifier)]
#![feature(label_break_value)]
#![feature(nll)]
#![feature(try_trait)]
#![feature(slice_patterns)]
#![feature(unicode_internals)]

#![recursion_limit="256"]

pub use errors;
use rustc_data_structures::sync::Lock;
use rustc_index::bit_set::GrowableBitSet;
pub use rustc_data_structures::thin_vec::ThinVec;
use ast::AttrId;
use syntax_pos::edition::Edition;

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
    fn new(edition: Edition) -> Globals {
        Globals {
            // We have no idea how many attributes there will be, so just
            // initiate the vectors with 0 bits. We'll grow them as necessary.
            used_attrs: Lock::new(GrowableBitSet::new_empty()),
            known_attrs: Lock::new(GrowableBitSet::new_empty()),
            syntax_pos_globals: syntax_pos::Globals::new(edition),
        }
    }
}

pub fn with_globals<F, R>(edition: Edition, f: F) -> R
    where F: FnOnce() -> R
{
    let globals = Globals::new(edition);
    GLOBALS.set(&globals, || {
        syntax_pos::GLOBALS.set(&globals.syntax_pos_globals, f)
    })
}

pub fn with_default_globals<F, R>(f: F) -> R
    where F: FnOnce() -> R
{
    with_globals(edition::DEFAULT_EDITION, f)
}

scoped_tls::scoped_thread_local!(pub static GLOBALS: Globals);

#[macro_use]
pub mod diagnostics {
    #[macro_use]
    pub mod macros;
}

pub mod util {
    pub mod classify;
    pub mod comments;
    pub mod lev_distance;
    pub mod literal;
    pub mod node_count;
    pub mod parser;
    pub mod map_in_place;
}

pub mod ast;
pub mod attr;
pub mod expand;
pub use syntax_pos::source_map;
pub mod entry;
pub mod feature_gate {
    mod check;
    pub use check::{check_crate, check_attribute, get_features, feature_err, feature_err_issue};
}
pub mod mut_visit;
pub mod ptr;
pub mod show_span;
pub use syntax_pos::edition;
pub use syntax_pos::symbol;
pub mod sess;
pub mod token;
pub mod tokenstream;
pub mod visit;

pub mod print {
    pub mod pp;
    pub mod pprust;
    mod helpers;
}

pub mod early_buffered_lints;

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in librustc.
pub trait HashStableContext: syntax_pos::HashStableContext {}
