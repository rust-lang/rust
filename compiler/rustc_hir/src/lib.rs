//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

#![feature(array_value_iter)]
#![feature(crate_visibility_modifier)]
#![feature(const_fn)] // For the unsizing cast on `&[]`
#![feature(const_panic)]
#![feature(in_band_lifetimes)]
#![feature(iterator_fold_self)]
#![feature(once_cell)]
#![feature(or_patterns)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc_macros;

#[macro_use]
extern crate rustc_data_structures;

mod arena;
pub mod def;
pub mod definitions;
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

pub use hir::*;
pub use hir_id::*;
pub use lang_items::{LangItem, LanguageItems};
pub use stable_hash_impls::HashStableContext;
pub use target::{MethodKind, Target};
