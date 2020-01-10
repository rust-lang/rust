//! The Rust parser and macro expander.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/", test(attr(deny(warnings))))]
#![feature(bool_to_option)]
#![feature(box_syntax)]
#![feature(const_fn)]
#![feature(const_transmute)]
#![feature(crate_visibility_modifier)]
#![feature(label_break_value)]
#![feature(nll)]
#![feature(try_trait)]
#![feature(slice_patterns)]
#![feature(unicode_internals)]
#![recursion_limit = "256"]

use ast::AttrId;
use rustc_data_structures::sync::Lock;
use rustc_index::bit_set::GrowableBitSet;
use rustc_span::edition::{Edition, DEFAULT_EDITION};

#[macro_export]
macro_rules! unwrap_or {
    ($opt:expr, $default:expr) => {
        match $opt {
            Some(x) => x,
            None => $default,
        }
    };
}

pub struct Globals {
    used_attrs: Lock<GrowableBitSet<AttrId>>,
    known_attrs: Lock<GrowableBitSet<AttrId>>,
    rustc_span_globals: rustc_span::Globals,
}

impl Globals {
    fn new(edition: Edition) -> Globals {
        Globals {
            // We have no idea how many attributes there will be, so just
            // initiate the vectors with 0 bits. We'll grow them as necessary.
            used_attrs: Lock::new(GrowableBitSet::new_empty()),
            known_attrs: Lock::new(GrowableBitSet::new_empty()),
            rustc_span_globals: rustc_span::Globals::new(edition),
        }
    }
}

pub fn with_globals<R>(edition: Edition, f: impl FnOnce() -> R) -> R {
    let globals = Globals::new(edition);
    GLOBALS.set(&globals, || rustc_span::GLOBALS.set(&globals.rustc_span_globals, f))
}

pub fn with_default_globals<R>(f: impl FnOnce() -> R) -> R {
    with_globals(DEFAULT_EDITION, f)
}

scoped_tls::scoped_thread_local!(pub static GLOBALS: Globals);

pub mod util {
    pub mod classify;
    pub mod comments;
    pub mod lev_distance;
    pub mod literal;
    pub mod map_in_place;
    pub mod node_count;
    pub mod parser;
}

pub mod ast;
pub mod attr;
pub mod entry;
pub mod expand;
pub mod feature_gate {
    mod check;
    pub use check::{check_attribute, check_crate, feature_err, feature_err_issue, get_features};
}
pub mod mut_visit;
pub mod ptr;
pub mod show_span;
pub use rustc_session::parse as sess;
pub mod token;
pub mod tokenstream;
pub mod visit;

pub mod print {
    mod helpers;
    pub mod pp;
    pub mod pprust;
}

pub mod early_buffered_lints;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in librustc.
pub trait HashStableContext: rustc_span::HashStableContext {
    fn hash_attr(&mut self, _: &ast::Attribute, hasher: &mut StableHasher);
}

impl<AstCtx: crate::HashStableContext> HashStable<AstCtx> for ast::Attribute {
    fn hash_stable(&self, hcx: &mut AstCtx, hasher: &mut StableHasher) {
        hcx.hash_attr(self, hasher)
    }
}
