//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

#![feature(associated_type_defaults)]
#![feature(const_btree_new)]
#![feature(crate_visibility_modifier)]
#![feature(once_cell)]
#![feature(min_specialization)]
#![feature(never_type)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc_macros;

#[macro_use]
extern crate rustc_data_structures;

mod arena;
pub mod def;
pub mod def_path_hash_map;
pub mod definitions;
pub mod diagnostic_items;
pub use rustc_span::def_id;
mod hir;
pub mod hir_id;
pub mod intravisit;
pub mod itemlikevisit;
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
