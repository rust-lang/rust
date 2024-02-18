//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

#![feature(associated_type_defaults)]
#![feature(closure_track_caller)]
#![feature(let_chains)]
#![cfg_attr(bootstrap, feature(min_specialization))]
#![feature(never_type)]
#![feature(rustc_attrs)]
#![feature(variant_count)]
#![allow(internal_features)]

#[macro_use]
extern crate rustc_macros;

#[macro_use]
extern crate tracing;

#[macro_use]
extern crate rustc_data_structures;

extern crate self as rustc_hir;

mod arena;
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
mod stable_hash_impls;
mod target;
pub mod weak_lang_items;

#[cfg(test)]
mod tests;

pub use hir::*;
pub use hir_id::*;
pub use lang_items::{LangItem, LanguageItems};
pub use stable_hash_impls::HashStableContext;
pub use target::{MethodKind, Target};

arena_types!(rustc_arena::declare_arena);
