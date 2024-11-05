//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

// tidy-alphabetical-start
#![allow(internal_features)]
#![feature(associated_type_defaults)]
#![feature(closure_track_caller)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(rustc_attrs)]
#![feature(variant_count)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

extern crate self as rustc_hir;

mod arena;
pub mod attribute;
pub mod def;
pub mod def_path_hash_map;
pub mod definitions;
pub mod diagnostic_items;
pub use rustc_span::def_id;
mod hir;
pub mod hir_id;
pub mod intravisit;
pub mod lang_items;
pub mod pat_util;
mod stability;
mod stable_hash_impls;
mod target;
mod version;
pub use version::RustcVersion;
pub mod weak_lang_items;

#[cfg(test)]
mod tests;

pub use attribute::*;
#[doc(no_inline)]
pub use hir::*;
pub use hir_id::*;
pub use lang_items::{LangItem, LanguageItems};
pub use stability::*;
pub use stable_hash_impls::HashStableContext;
pub use target::{MethodKind, Target};

arena_types!(rustc_arena::declare_arena);
