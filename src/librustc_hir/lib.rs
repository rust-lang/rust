//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

#![feature(crate_visibility_modifier)]
#![feature(const_if_match)]
#![feature(const_fn)] // For the unsizing cast on `&[]`
#![feature(const_panic)]
#![feature(in_band_lifetimes)]
#![feature(specialization)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc_data_structures;

pub mod def;
pub use rustc_span::def_id;
mod hir;
pub mod hir_id;
pub mod intravisit;
pub mod itemlikevisit;
pub mod lang_items;
pub mod pat_util;
pub mod print;
mod stable_hash_impls;
mod target;
pub mod weak_lang_items;

pub use hir::*;
pub use hir_id::*;
pub use lang_items::{LangItem, LanguageItems};
pub use stable_hash_impls::HashStableContext;
pub use target::{MethodKind, Target};
