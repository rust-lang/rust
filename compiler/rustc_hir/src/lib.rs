//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

// tidy-alphabetical-start
#![feature(associated_type_defaults)]
#![feature(closure_track_caller)]
#![feature(debug_closure_helpers)]
#![feature(exhaustive_patterns)]
#![feature(never_type)]
#![feature(variant_count)]
#![recursion_limit = "256"]
// tidy-alphabetical-end

extern crate self as rustc_hir;

mod arena;
pub mod attrs;
pub mod def;
pub mod def_path_hash_map;
pub mod definitions;
pub mod diagnostic_items;
pub use rustc_span::def_id;
mod hir;
pub use rustc_hir_id::{self as hir_id, *};
pub mod intravisit;
pub mod lang_items;
pub mod limit;
pub mod lints;
pub mod pat_util;
mod stability;
mod stable_hash_impls;
mod target;
mod version;
pub mod weak_lang_items;

#[cfg(test)]
mod tests;

#[doc(no_inline)]
pub use hir::*;
pub use lang_items::{LangItem, LanguageItems};
pub use stability::*;
pub use stable_hash_impls::HashStableContext;
pub use target::{MethodKind, Target};
pub use version::*;

arena_types!(rustc_arena::declare_arena);
