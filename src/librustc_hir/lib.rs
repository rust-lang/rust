//! HIR datatypes. See the [rustc guide] for more info.
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/hir.html

#![feature(crate_visibility_modifier)]
#![feature(const_fn)] // For the unsizing cast on `&[]`
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
pub mod pat_util;
pub mod print;
mod stable_hash_impls;
pub use hir::*;
pub use hir_id::*;
pub use stable_hash_impls::HashStableContext;
